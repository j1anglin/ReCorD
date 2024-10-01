import os
import cv2
import json
import numpy as np
import mediapipe as mp
from mediapipe import solutions
import matplotlib.pyplot as plt
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

# ==================================================================================================
# Reference:
# https://developers.google.com/mediapipe/solutions/vision/pose_landmarker/python
# https://github.com/googlesamples/mediapipe/blob/main/examples/pose_landmarker/python/%5BMediaPipe_Python_Tasks%5D_Pose_Landmarker.ipynb
# https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/pose_landmarker/python/%5BMediaPipe_Python_Tasks%5D_Pose_Landmarker.ipynb#scrollTo=_JVO3rvPD4RN

# ==================================================================================================


def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style())
    return annotated_image


# Initialize pose keypoints
body_keypoints = {
    0: "Nose",
    1: "Neck",
    2: "RShoulder",
    3: "RElbow",
    4: "RWrist",
    5: "LShoulder",
    6: "LElbow",
    7: "LWrist",
    8: "RHip",
    9: "RKnee",
    10: "RAnkle",
    11: "LHip",
    12: "LKnee",
    13: "LAnkle",
    14: "REye",
    15: "LEye",
    16: "REar",
    17: "LEar"
}

# image_path = "./hico_20160224_det/images/train2015/HICO_train2015_00000107.jpg"
# image_path = "./hico_20160224_det/images/train2015/HICO_train2015_00000127.jpg"
# raw_image = cv2.imread(image_path)
txt_name="carry_bike"
f=open(txt_name+".txt", 'r')
images_names = []
for line in f:
    images_names.append(line.strip())
print(len(images_names))

# STEP 2: Create an PoseLandmarker object.
base_options = python.BaseOptions(model_asset_path='./pose_landmarker.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)

# STEP 3: Load the input image.
# image = mp.Image.create_from_file(image_path)
output_dir="./pose_landmark/"+txt_name
os.makedirs(output_dir, exist_ok=True)


# STEP 4: Detect pose landmarks from the input image.
# detection_result = detector.detect(image)
# print(len(detection_result))
# pose_landmarker_result = landmarker.detect(mp_image)

# STEP 5: Process the detection result. In this case, visualize it.
# annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
# plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
# plt.savefig("pose_landmark.jpg")
# plt.show()
normalized_landmark_list = []
for image_name in images_names:
    if "train2015" in image_name:
        image_path=f"./hico_20160224_det/images/train2015/{image_name}"
    elif "test2015" in image_name:
        image_path=f"./hico_20160224_det/images/test2015/{image_name}"
    else:
        print(f"Invalid image name: {image_name}")
        continue
    img=cv2.imread(image_path)
    print(img.shape)
    image = mp.Image.create_from_file(image_path)
    detection_result = detector.detect(image)
    normalized_kp_coords=detection_result.pose_landmarks[0]
    kp_coords=[]
    for norm_kp_coords in normalized_kp_coords:
        landmark_x = norm_kp_coords.x*img.shape[1]
        landmark_y = norm_kp_coords.y*img.shape[0]
        kp_coords.append([landmark_x, landmark_y])
    normalized_landmark_list.append([image_name, img.shape, kp_coords])
    # print(f"Processed image: {image_name}")
    # print(f"Number of keypoints: {len(kp_coords)}")
    # print(f"Keypoints: {kp_coords}")
# print(f"Keypoints: {normalized_landmark_list}")
json.dump(normalized_landmark_list, open(f"{output_dir}/keypoints.json", 'w'))
