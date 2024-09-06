import random
from p2pp import p2pp
from banks import Prompt


"""
https://github.com/masci/banks
banks s the linguist professor who will help you generate meaningful LLM prompts using a template language that makes sense.
"""


prompt_template = """
# Your Role: Expert Human Pose Analyst

## Objective: Think step by step, your task is analyzing keypoints of human pose in square images according to the user's prompt and manipulating the bounding boxes of the object to the correct locations while maintaining visual accuracy.

## Human Pose Keypoints and Bounding Box Specifications and Analysis
1. Image Coordinate: Define square images with top-left at [0, 0] and bottom-right at [512, 512].
2. Annotations of Keypoints: ["nose", "left eye inner", "left eye", "left eye outer", "right eye inner", "right eye", "right eye outer", "left ear", "right ear", "mouth left", "mouth right", "left shoulder", "right shoulder", "left elbow", "right elbow", "left wrist", "right wrist", "left pinky", "right pinky", "left index", "right index", "left thumb", "right thumb", "left hip", "right hip", "left knee", "right knee", "left ankle", "right ankle", "left heel", "right heel", "left foot index", "right foot index"]
3. Box Format: [Top-left x, Top-left y, Bottom-right x, Bottom-right y]
4. Object Size: The object's bounding box size is represented as a fraction of the image size.
5. Results of Analysis: Pose Types: ["Static", "Dynamic"], Body Orientation: ["Frontal", "Backward", "Profile", "Angled"], Facial Direction: ["Directly at Viewer", "Looking Upwards", "Looking Downwards", "Looking Sideways", "Looking at objects"], Object Relationship: ["Above human", "Under human", "Beside human", "In front of human", "Behind human", "On human", "Near human"], Object Size: ["one-tenth", "one-fifth", "three-tenths", "two-fifths", "one-half", "three-fifths", "seven-tenths", "four-fifths", "nine-tenths", "one"], Object Location: [,,,]

## Key Guidelines
1. Alignment: Follow the user's prompt, keeping the attributes of the specified object.
2. Boundary Adherence: Keep the bounding box coordinate within [0, 512].
3. Visual Accuracy: Ensure the object's bounding box size is visually accurate and aligned with the human pose.
4. Minimal Modifications: Change the bounding box of the object only if it doesn't match the scene affordances.
5. Human Location Constraints: The human bounding box should not be altered.
6. Overlap Reduction: Minimize intersections of all the bounding boxes.

## Process Steps
1. Interpret Prompts: Read and understand the user's prompt.
2. Keypoints Analysis: Identify all keypoints of the person and perform pose estimation to understand the spatial relationships between different keypoints.
3. Implement Changes: Review and adjust the current bounding box of object while considering the interaction and scene affordances.
4. Explain Adjustments: Justify the reasons behind the alteration and ensure the adjustment abides by the key guidelines.
5. Output the Results: Present the analysis and predict the updated absolute coordinates of the object's bounding box, which should include a list of bounding boxes in Python format.

## Examples

- Example 1
    User Prompt: a woman is sitting on a horse. Keypoints: [[228, 87], [232, 81], [234, 81], [235, 82], [228, 80], [228, 79], [227, 78], [242, 83], [232, 80], [232, 95], [228, 94], [254, 118], [233, 113], [264, 173], [234, 163], [236, 202], [223, 194], [233, 219], [219, 200], [227, 211], [217, 201], [230, 206], [220, 200], [264, 201], [243, 200], [233, 246], [206, 263], [239, 309], [169, 373], [244, 331], [168, 392], [226, 352], [145, 396]], Original Human Location: [200, 43, 278, 358], Original Object Location: [120, 46, 452, 389].
    Reasoning: Here, there is one woman and one horse.
    Pose Types: "Dynamic"
    Body Orientation: "Profile"
    Facial Direction: "Looking Sideways"
    Object Relationship: "Under human"
    Object Size: "two-fifths"
    Updated Object Location: [101, 131, 433, 474]

- Example 2
    User Prompt: a baby is carrying a teddy bear. Keypoints: [[325, 181], [345, 157], [361, 156], [371, 156], [299, 161], [286, 162], [275, 162], [389, 161], [253, 168], [349, 202], [298, 209], [408, 234], [256, 227], [390, 327], [275, 312], [272, 303], [376, 315], [231, 308], [400, 328], [237, 282], [408, 308], [245, 280], [402, 303], [388, 370], [301, 365], [387, 447], [239, 438], [366, 523], [217, 521], [359, 537], [226, 541], [355, 562], [187, 552]], Original Human Location: [9, 44, 486, 507], Original Object Location: [7, 28, 486, 446].
    Reasoning: Here, there is one baby and one teddy bear.
    Pose Types: "Static"
    Body Orientation: "Angled"
    Facial Direction: "Looking Sideways"
    Object Relationship: "On human"
    Object Size: "four-fifths"
    Updated Object Location: [0, 86, 479, 504]

- Example 3
    User Prompt: a man is driving a taxi. Keypoints: [[177, 173], [178, 167], [180, 167], [181, 166], [173, 167], [171, 166], [169, 167], [182, 167], [166, 169], [180, 178], [174, 179], [189, 189], [160, 198], [204, 222], [185, 227], [191, 227], [202, 224], [186, 232], [206, 228], [185, 225], [206, 222], [187, 223], [204, 220], [182, 272], [164, 275], [202, 312], [185, 321], [202, 369], [172, 377], [199, 381], [168, 387], [209, 380], [179, 389]], Original Human Location: [143, 130, 220, 244], Original Object Location: [7, 17, 468, 442].
    Reasoning: Here, there is one man and one taxi.
    Pose Types: "Static"
    Body Orientation: "Angled"
    Facial Direction: "Directly at Viewer"
    Object Relationship: "Near human"
    Object Size: "seven-tenths"
    Updated Object Location: [33, 85, 494, 510]

Your Current Task: Carefully follow the provided guidelines and steps closely to accurately identify the human pose based on the given prompt and adjust the bounding boxes in accordance with the user's prompt. Ensure adherence to the above output format.

User Prompt: { {{ GT_prompt }}. Keypoints: {{ kps }}, Original Human Location: {{ ori_h_bbox }}, Original Object Location: {{ ori_o_bbox }}.}
Reasoning: 
"""

# Sample Code
# p = Prompt(prompt_template)
# GT_prompt = "a man is chasing a bird"
# kps = [[258, 156], [259, 148], [258, 147], [256, 146], [261, 148], [261, 148], [261, 148], [247, 140], [254, 144], [252, 162], [254, 162], [194, 161], [236, 168], [166, 210], [250, 220], [190, 236], [272, 211], [194, 242], [279, 209], [202, 243], [275, 201], [202, 242], [272, 202], [179, 270], [207, 275], [192, 332], [269, 313], [154, 402], [217, 370], [138, 417], [202, 378], [173, 418], [239, 407]]
# ori_h_bbox = [96, 13, 265, 351]
# ori_o_bbox = [365, 99, 453, 229]
# print(p.text({"GT_prompt": "a man is chasing a bird", "kps": kps, "ori_h_bbox": ori_h_bbox, "ori_o_bbox": ori_o_bbox}))


class PromptTemplate:
    def __init__(self, GT_prompt: str = "", kps: list = [], ori_h_bbox: list = [], ori_o_bbox: list = []) -> None:
        self.GT_prompt = GT_prompt
        self.kps = kps
        self.ori_h_bbox = ori_h_bbox
        self.ori_o_bbox = ori_o_bbox

    def get(self):
        # Format the prompt
        p = Prompt(prompt_template)
        return (p.text({"GT_prompt": self.GT_prompt, "kps": self.kps, "ori_h_bbox": self.ori_h_bbox, "ori_o_bbox": self.ori_o_bbox}))

def generate_subject():
    subjects = ["man", "woman", "boy", "girl", "old man",
                "old woman", "teenager", "child", "young man", "young woman",
                "adult", "kid", "elderly person", "middle-aged person", "toddler"]
    return random.choice(subjects)

def formatting(input_verb, object):
    random_subject = generate_subject()
    # random_subject = "person"
    
    def determine_article_capital(word):
        return "An" if word[0].lower() in "aeiou" else "A"
    
    def determine_article(word):
        return "an" if word[0].lower() in "aeiou" else "a"
    subject_article = determine_article_capital(random_subject)
    object_article = determine_article(object)
    
    if "_" in input_verb:
        split_verb=input_verb.split("_")
        verb = split_verb[0]
        preposition = split_verb[1]
        verbing = p2pp(verb)
        GT_prompt=f"{subject_article} {random_subject} is {verbing} {preposition} {object_article} {object}"
    else:
        verb = p2pp(input_verb)
        GT_prompt=f"{subject_article} {random_subject} is {verb} {object_article} {object}"
    return GT_prompt