import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

from processors import *
from diffusers import DDIMScheduler
from config import RunConfig
from diffusers import AutoencoderKL
from run import run_on_prompt
import matplotlib.pyplot as plt
import torch
from prompt_to_prompt_pipeline import Prompt2PromptPipeline
from PIL import Image
from pathlib import Path
from tqdm import tqdm  
import pandas as pd
import ast
import json
import gc

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
#vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
if device.type == "cuda":
    pipe = Prompt2PromptPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", 
                                                torch_dtype=torch.float16, use_safetensors=True).to(device)
else:
    pipe = Prompt2PromptPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",
                                                torch_dtype=torch.float32, use_safetensors=False).to(device)

pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)





n_self_replace = 0.3
generate_imgs_num = 1

dataset = 'vcoco'
backbone = 'SDXL'
prompts_path = f'./prompts_{dataset}_w_seeds_{backbone}.csv'
output_path=Path(f'./data/{dataset}/final_images_{backbone}_03')



start_index = 0
end_index = 1275

df = pd.read_csv(prompts_path)
for index, row in tqdm(df.iloc[start_index:end_index].iterrows(), total=(end_index-start_index)):

    seeds = [ast.literal_eval(row['seeds'])[int(row['selected_image_idx'])]]
    prompts = [ row['Intransitive_prompt'], row['full_prompt']]
    
    cross_attention_kwargs = {"edit_type": "refine",
                              "n_self_replace": n_self_replace,
                              "n_cross_replace": {"default_": 1.0, f"{row['obj_nouns']}": 0.},
                              }
    config = RunConfig(prompt=prompts,
                       guidance_scale = 7.5,
                       n_inference_steps = 50,                       
                       run_standard_sd=False,
                       scale_factor=20,
                       max_iter_to_alter=25,
                       output_path=output_path,
                       viz_path=Path('./attention_maps/hicodet/noisy_images'),
                       pred_box=True,
                       viz_attention=False
                       
                       )

    indices_to_alter = [row['obj_index']]
    indices_to_viz = [row['subject_index'],  row['verb_index']]

    if pd.isnull(row[f'updated_object_bbox']):
        bbox = []
    else:
        x1, y1, x2, y2 = ast.literal_eval(row[f'updated_object_bbox'])
        if x1 > x2 or y1 > y2: 
            bbox = []
        else:
            x1, y1 = min(x1, 512), min(y1, 512)
            x2, y2 = min(x2, 512), min(y2, 512)
            bbox = [[x1, y1, x2, y2]]
        
    if os.path.exists(config.output_path / prompts[1] / '0.png'):  
        continue 


    for i, seed in enumerate(seeds):

        g_cpu = torch.Generator().manual_seed(seed)
        image, object_bbox = run_on_prompt(
                                         model = pipe,
                                         prompts=prompts, 
                                         cross_attention_kwargs=cross_attention_kwargs, 
                                         generator=g_cpu,
                                         indices_to_alter=indices_to_alter,
                                         indices_to_viz=indices_to_viz,
                                         bbox=bbox,

                                         scale_factor=config.scale_factor,
                                         max_iter_to_alter=config.max_iter_to_alter,
                                         config=config
                                        )
                                        
        print(f"object_bbox: {object_bbox}")
        img = image['images'][1]
        img = img.resize((512, 512), Image.LANCZOS)
        
        
        prompt_dir_path = config.output_path / prompts[1]
        prompt_dir_path.mkdir(exist_ok=True, parents=True)
        img_path = prompt_dir_path / f'{i}.png'
        img.save(img_path)
        
        obj_bbox_dir_path = Path(f'./data/{dataset}/obj_bbox_jsons/') / prompts[1]
        obj_bbox_dir_path.mkdir(exist_ok=True, parents=True)

        obj_bbox_save_path = obj_bbox_dir_path / f'{i}.json'
        data = {'object_bbox': object_bbox}
        with open(obj_bbox_save_path, 'w') as json_file:
            json.dump(data, json_file) 
        

        del image, object_bbox  
        gc.collect()
        torch.cuda.empty_cache()
        """
        for i, img in enumerate(image['images']):
            prompt_dir_path = config.output_path / prompts[1]
            # Ensure the directory for the prompt exists
            prompt_dir_path.mkdir(exist_ok=True, parents=True)

            img_path = prompt_dir_path / f'{i}.png'
            img.save(img_path)
            print(f'Prompt: {prompts[1]}, image {i} saved to {img_path}')
        """