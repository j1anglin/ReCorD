import ast
import random


def shift_coordinates(bbox_string, max_width, max_height):
    # Convert string representation to a tuple of integers
    coordinates = ast.literal_eval(bbox_string)

    x_min, y_min, x_max, y_max = coordinates
    box_width, box_height = x_max - x_min, y_max - y_min

    # Calculate maximum shift without resizing the box
    shift_x = random.randint(-x_min, max_width - x_max)
    shift_y = random.randint(-y_min, max_height - y_max)

    # Return the shifted coordinates as a list
    return [x_min + shift_x, y_min + shift_y, x_min + shift_x + box_width, y_min + shift_y + box_height]


def shift_keypoints(keypoints, max_width, max_height):
    # Find the bounding box of the keypoints
    x_coords = [kp[0] for kp in keypoints]
    y_coords = [kp[1] for kp in keypoints]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    # Calculate the maximum and minimum possible shifts
    max_shift_x = max_width - x_max
    min_shift_x = -x_min
    max_shift_y = max_height - y_max
    min_shift_y = -y_min

    # Ensure that shift ranges are valid
    shift_x = random.randint(min_shift_x, max_shift_x) if min_shift_x < max_shift_x else 0
    shift_y = random.randint(min_shift_y, max_shift_y) if min_shift_y < max_shift_y else 0

    # Apply the shift to each keypoint
    shifted_keypoints = [[x + shift_x, y + shift_y] for x, y in keypoints]
    return shifted_keypoints


# Sample usage
# human_bboxes = "[169, 20, 349, 379]"
# keypoints = [[258, 156], [259, 148], [258, 147], [256, 146], [261, 148], [261, 148], [261, 148], [247, 140], [254, 144], [252, 162], [254, 162], [194, 161], [236, 168], [166, 210], [250, 220], [190, 236], [
#     272, 211], [194, 242], [279, 209], [202, 243], [275, 201], [202, 242], [272, 202], [179, 270], [207, 275], [192, 332], [269, 313], [154, 402], [217, 370], [138, 417], [202, 378], [173, 418], [239, 407]]
# max_width, max_height = 512, 512  # Dimensions of the image
# shifted_bboxes = shift_coordinates(human_bboxes, max_width, max_height)
# print(shifted_bboxes)
# shifted_keypoints = shift_keypoints(keypoints, max_width, max_height)
# print(shifted_keypoints)
