import random
import numpy as np
import mediapipe as mp

from utils.visualize import plot_bounding_boxes
from utils.prompt import PromptTemplate, formatting
from utils.data_utils import read_json, encode_image
from utils.api import gpt4v_single_img, gpt4v_multi_img
from utils.pose import pose_landmarker, find_human_bboxes
from utils.response import parse_bboxes, parse_image_number
from utils.transform import shift_coordinates, shift_keypoints


def pose_selection(generated_image_path_list, api_key, prompt, input_annotations=None):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    output = gpt4v_multi_img(headers, generated_image_path_list, prompt)
    selected_image_idx = parse_image_number(output)
    print(f"Selected image index: {selected_image_idx}")
    return selected_image_idx


def interaction_aware_reasoning(generated_image_path, api_key, input_annotations=None):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    # Create a pose landmark detector
    detector = pose_landmarker()

    verb = input_annotations['verb']
    object = input_annotations['object']
    # human_bboxes = input_annotations['human_bboxes']
    object_bboxes = input_annotations['object_bboxes']

    if verb == "no_interaction":
        return

    GT_prompt = formatting(verb, object)
    image = mp.Image.create_from_file(generated_image_path)
    detection_result = detector.detect(image)

    if len(detection_result.pose_landmarks) > 0:
        print(f"Pose landmarks detected in {generated_image_path}")
        print(f"GT Prompt: {GT_prompt}")
        seg_mask = detection_result.segmentation_masks[0].numpy_view()
        seg_mask = (seg_mask*255).astype(np.uint8)
        human_bboxes = find_human_bboxes(seg_mask)
        print("human_bboxes: ", human_bboxes)
        normalized_kp_coords = detection_result.pose_landmarks[0]
        scaled_kp_coords = []
        for norm_kp_coords in normalized_kp_coords:
            kp_coords_x = norm_kp_coords.x*512
            kp_coords_y = norm_kp_coords.y*512
            scaled_kp_coords.append(
                [int(kp_coords_x), int(kp_coords_y)])
        # print(scaled_kp_coords)
        p = PromptTemplate(GT_prompt, scaled_kp_coords,
                           human_bboxes, object_bboxes)
        base64_image = encode_image(generated_image_path)
        output = gpt4v_single_img(headers, base64_image, p)
        updated_object_location = parse_bboxes(output, object)
        plot_bounding_boxes(generated_image_path, updated_object_location,
                            GT_prompt, human_bboxes, object_bboxes)
        return updated_object_location
    else:
        print(f"No pose landmarks detected in {generated_image_path}")
        return
