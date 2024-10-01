from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def pose_landmarker():
    # Create an PoseLandmarker object.
    base_options = python.BaseOptions(
        model_asset_path='./GPT4V/pose/pose_landmarker.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        output_segmentation_masks=True)
    detector = vision.PoseLandmarker.create_from_options(options)
    return detector

def find_human_bboxes(pixel_list):
    top, bottom, left, right = None, None, None, None

    # Iterate through each row
    for i in range(len(pixel_list)):
        # Iterate through each column
        for j in range(len(pixel_list[i])):
            # Check if the pixel value is non-zero
            if pixel_list[i][j] != 0:
                # Update top index
                if top is None or i < top[0]:
                    top = (i, j)
                # Update bottom index
                if bottom is None or i > bottom[0]:
                    bottom = (i, j)
                # Update left index
                if left is None or j < left[1]:
                    left = (i, j)
                # Update right index
                if right is None or j > right[1]:
                    right = (i, j)

    return [left[1], top[0], right[1], bottom[0]]
