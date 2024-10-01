import json
import random

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def scale_coordinates(coordinates, scale_x, scale_y):
    x_min, y_min, x_max, y_max = coordinates
    return [int(x_min * scale_x), int(y_min * scale_y), int(x_max * scale_x), int(y_max * scale_y)]

def shift_coordinates(coordinates, max_width, max_height):
    x_min, y_min, x_max, y_max = coordinates
    box_width, box_height = x_max - x_min, y_max - y_min

    # Calculate maximum shift without resizing the box
    shift_x = random.randint(-x_min, max_width - x_max)
    shift_y = random.randint(-y_min, max_height - y_max)
    return [x_min + shift_x, y_min + shift_y, x_min + shift_x + box_width, y_min + shift_y + box_height]

def update_json_data(json_list, new_width, new_height):
    updated_list=[]
    for json_data in json_list:
        if "test" in json_data['global_id']:
            continue
        print(json_data['global_id'])
        original_width , original_height, _= json_data['image_size']
        # print(original_width, original_height)
        scale_x = new_height / original_height
        scale_y = new_width / original_width
        # print(scale_x, scale_y)

        for hoi in json_data['hois']:
            # Scale coordinates
            hoi['human_bboxes']=[scale_coordinates(human_bbox, scale_x, scale_y) for human_bbox in hoi['human_bboxes']]
            hoi['object_bboxes']=[scale_coordinates(object_bbox, scale_x, scale_y) for object_bbox in hoi['object_bboxes']]
            json_data['image_size']=[new_width, new_height, 3]
            # Shift coordinates
            # hoi['human_bboxes_shifted']=[shift_coordinates(human_bbox, new_width, new_height) for human_bbox in hoi['human_bboxes']]
            hoi['object_bboxes_shifted']=[shift_coordinates(object_bbox, new_width, new_height) for object_bbox in hoi['object_bboxes']]
        updated_list.append(json_data)    
    return updated_list

def save_json(file_path, json_data):
    with open(file_path, 'w') as f:
        json.dump(json_data, f, indent=4)

file_path='./hico_20160224_det_processed/anno_list.json'
json_data = load_json(file_path)
updated_json_data = update_json_data(json_data, 512, 512)

save_json('./hico_20160224_det_processed/anno_list_512_shifted.json', updated_json_data)