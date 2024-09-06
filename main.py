# +
from processors import *
from diffusers import DDIMScheduler
from config import RunConfig
from diffusers import AutoencoderKL
from run import run_on_prompt, split_sentence
import matplotlib.pyplot as plt
import torch
from record_pipeline import ReCorDPipeline
import os
from PIL import Image
from pathlib import Path
    
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

if device.type == "cuda":
    pipe = ReCorDPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", 
                                                torch_dtype=torch.float16, use_safetensors=True).to(device)
else:
    pipe = ReCorDPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",
                                                torch_dtype=torch.float32, use_safetensors=False).to(device)

pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

prompts = ['a boy is feeding', 'a boy is feeding a bird']

obj_nouns = "a bird"
subject_index = 2
obj_index  = 6
bbox = [ [0, 300, 200, 400] ]
# bbox = [ ]

cross_attention_kwargs = {"edit_type": "ReCorD",
                            "n_self_replace": 0.2,
                            "n_cross_replace": {"default_": 1.0, f"{obj_nouns}": 0.},
                            }
config = RunConfig(prompt=prompts,
                    guidance_scale = 7.5,
                    n_inference_steps = 50,                       
                    run_standard_sd=False,
                    scale_factor=20,
                    max_iter_to_alter=25,
                    
                    output_path=Path('./outputs/exp'),
                    viz_path=Path('./attention_maps/exp')
                    )

seeds = [2114]

indices_to_alter = [obj_index]
indices_to_viz = [subject_index, subject_index+1]


for i, seed in enumerate(seeds):

    g_cpu = torch.Generator().manual_seed(seed)
    image, obj_bbox = run_on_prompt(
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
    
    torch.cuda.empty_cache()
    
    for j, img in enumerate(image['images']):
        prompt_dir_path = config.output_path / prompts[1]
        # Ensure the directory for the prompt exists
        prompt_dir_path.mkdir(exist_ok=True, parents=True)

        img_path = prompt_dir_path / f'seed_{seed}_{j}.png'
        img.save(img_path)
        print(f'Prompt: {prompts[1]}, image {i} saved to {img_path}')

        
