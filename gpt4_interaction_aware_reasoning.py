from GPT4V.GPT4V.gpt4v import pose_selection, interaction_aware_reasoning
from tqdm import tqdm  
import pandas as pd
import ast
import json
from pathlib import Path
import os
import ast

def update_csv_and_original_obj_bbox(df, dataset, backbone):
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        df.at[index, 'selected_image_idx'] = str(int(row["selected_image_idx"]))

        obj_bbox_json_path = obj_bbox_json_root / row['full_prompt'] / f"{int(row['selected_image_idx'])}.json"
        with obj_bbox_json_path.open('r') as file:
            data = json.load(file)
            df.at[index, 'object_bbox'] = str(data["object_bbox"])
        
    return df

def update_csv_and_updated_obj_bbox(df, dataset, backbone):
    ct = 0
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        reasoning_json_path = Path(f"./data/{dataset}/reasoning_jsons_{backbone}/{row['full_prompt']}") / "reasoning_output.json"

        if os.path.exists(reasoning_json_path):
            with open(reasoning_json_path, 'r') as file:
                data = json.load(file)
                updated_object_location = data['updated_object_location']
                if updated_object_location != None:
                    df.at[index, 'updated_object_bbox'] = str(updated_object_location)
                    continue

        df.at[index, 'updated_object_bbox'] = None
        ct = ct + 1

    print(f"{ct} rows not update obj bbox")
    return df


dataset = 'vcoco'
backbone = 'SDXL'
csv_path = f'./prompts_{dataset}_w_seeds_{backbone}.csv'

api_key = ''

imgs_root = Path('./data/') / dataset / f'noisy_images_{backbone}'
obj_bbox_json_root = Path('./data/') / dataset / f'obj_bbox_jsons_{backbone}'
reasoning_json_root = Path('./data/')  / dataset / f'reasoning_jsons_{backbone}'


df = pd.read_csv(csv_path)
"""
df = update_csv_and_updated_obj_bbox(df, dataset, backbone)
df.to_csv(csv_path, index=False)

"""
start_index = 0
end_index = 2550


for index, row in tqdm(df.iloc[start_index:end_index].iterrows(), total=(end_index-start_index)):
    if pd.isnull(row['updated_object_bbox']):
        try:
            full_prompt = row['full_prompt']
            img_selected = row['selected_image_idx']

            generated_image_path =  str (imgs_root/ full_prompt / f'{img_selected}.png')
            input_annotations ={
                "object_bboxes" : ast.literal_eval(row['object_bbox']),
                "object" : row['obj_nouns'].split()[-1],
            }


            output, updated_object_location = interaction_aware_reasoning(generated_image_path, api_key, full_prompt, input_annotations)


            data = {'updated_object_location': updated_object_location,
                    'output': output["choices"][0]["message"] }

            reasoning_json_save_path = reasoning_json_root / full_prompt 
            reasoning_json_save_path.mkdir(exist_ok=True, parents=True)
            with open(reasoning_json_save_path / "reasoning_output.json", 'w') as json_file:
                json.dump(data, json_file)    
        except:
            pass
