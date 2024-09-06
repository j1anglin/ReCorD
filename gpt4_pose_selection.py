from GPT4V.GPT4V.gpt4v import pose_selection, interaction_aware_reasoning
from tqdm import tqdm  
import pandas as pd
import ast
import json
from pathlib import Path
import os

"""
def pose_selection(generated_image_path_list, api_key, prompt, input_annotations=None):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    output = gpt4v_multi_img(headers, generated_image_path_list, prompt)
    selected_image_idx = parse_image_number(output)
    print(f"Selected image index: {selected_image_idx}")
    return selected_image_idx
"""

dataset = 'hicodet'
backbone = 'SDXL'
csv_path = f'./prompts_{dataset}_w_seeds_{backbone}.csv'
api_key = ''

imgs_root = Path('./data') / dataset / f'noisy_images_{backbone}'
pose_sel_json_root = Path('./data') / dataset / f'pose_selection_{backbone}'

df = pd.read_csv(csv_path)
start_index = 0
end_index = 1000


for index, row in tqdm(df.iloc[start_index:end_index].iterrows(), total=(end_index-start_index)):
    full_prompt = row['full_prompt']
    generated_image_path_list = [ str(imgs_root/ full_prompt / f'{i}.png') for i in range(5) ]
    pose_sel_json_save_path = pose_sel_json_root / full_prompt 
    #try:
    output, selected_image_idx = pose_selection(generated_image_path_list, api_key, full_prompt)
    df.at[index, 'selected_image_idx'] = str(selected_image_idx)


    pose_sel_json_save_path.mkdir(exist_ok=True, parents=True)

    data = {'selected_image_idx': selected_image_idx,
            'output': output["choices"][0]["message"] }

    with open(pose_sel_json_save_path / "pose_sel_output.json", 'w') as json_file:
        json.dump(data, json_file)  

    if (selected_image_idx == None) :
        print(f"prompt: {row['full_prompt']}")  
    #except:
    #    pass    
    """
    if pd.isnull(row['selected_image_idx']):
        try:
        output, selected_image_idx = pose_selection(generated_image_path_list, api_key, full_prompt)
        df.at[index, 'selected_image_idx'] = str(selected_image_idx)


        pose_sel_json_save_path.mkdir(exist_ok=True, parents=True)

        data = {'selected_image_idx': selected_image_idx,
                'output': output["choices"][0]["message"] }

        with open(pose_sel_json_save_path / "pose_sel_output.json", 'w') as json_file:
            json.dump(data, json_file)  

        if (selected_image_idx == None) :
            print(f"prompt: {row['full_prompt']}")  
        except:
            pass
    """
"""
for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    full_prompt = row['full_prompt']
    generated_image_path_list = [ str(imgs_root/ full_prompt / f'{i}.png') for i in range(5) ]
    pose_sel_json_save_path = pose_sel_json_root / full_prompt 
    
    if os.path.exists(pose_sel_json_save_path):  
        continue 

    try:
        output, selected_image_idx = pose_selection(generated_image_path_list, api_key, full_prompt)
        df.at[index, 'selected_image_idx'] = str(selected_image_idx)

        
        pose_sel_json_save_path.mkdir(exist_ok=True, parents=True)

        data = {'selected_image_idx': selected_image_idx,
                'output': output["choices"][0]["message"] }


        with open(pose_sel_json_save_path / "pose_sel_output.json", 'w') as json_file:
            json.dump(data, json_file)    

"""