import re

def parse_image_number(api_response):
    content = api_response["choices"][0]["message"]["content"]
    # Regular expressions to match "second" or "number 2"
    ordinal_map = {
        'first': 0, '1st': 0, 'number 1': 0, 'one': 0,
        'second': 1, '2nd': 1, 'number 2': 1, 'two': 1,
        'third': 2, '3rd': 2, 'number 3': 2, 'three': 2,
        'fourth': 3, '4th': 3, 'number 4': 3, 'four': 3,
        'fifth': 4, '5th': 4, 'number 5': 4, 'five': 4,
        # Add more mappings as needed
    }
    
    # Combine all keys from the map into a single regex pattern
    pattern = r'\b(?:' + '|'.join(re.escape(key) for key in ordinal_map.keys()) + r')\b'
    matches = re.findall(pattern, content, re.IGNORECASE)
    
    # Return the zero-based index of the first match, if any
    if matches:
        # Convert the match to lower case to match the dictionary keys
        match = matches[0].lower()
        if match in ordinal_map:
            return ordinal_map[match]
    return None


def find_bbox(text, label):
    # Search for labeled bounding box first if label is provided
    if label:
        labeled_pattern = rf"Object Location \({label}\): \[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]"
        labeled_matches = re.findall(labeled_pattern, text)
        if labeled_matches:
            # Return the first match for the specific label
            return tuple(map(int, labeled_matches[0]))

    # Pattern to capture all instances of "Object Location", with or without specific labels
    all_pattern = r"Object Location(?: \([^\)]+\))?: \[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]|object_location: \[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]"
    all_matches = re.findall(all_pattern, text)

    # Process all matches to convert them to integer tuples
    processed_matches = []
    for match in all_matches:
        # Flatten the match groups to filter out empty strings and convert to integers
        processed_match = tuple(map(int, [m for m in match if m]))
        processed_matches.append(processed_match)

    # Return the last bounding box if any matches are found
    if processed_matches:
        return processed_matches[-1]

    # Return None if no matches are found
    return None


def parse_bboxes(api_response, object):
    content = api_response["choices"][0]["message"]["content"]

    # Search for human and object bounding box patterns
    # human_location = find_bbox(content, "Human")
    object_location = find_bbox(content, object)

    # if human_location:
    #     print("Suggested Human Location:", human_location)

    if object_location:
        print("Suggested Object Location:", object_location)

    return object_location


# Sample usage of parse_bboxes()
# api_response_1 = {'id': 'chatcmpl-8kv2cfashtOtXTDqjCX5uASGAaYEx', 'object': 'chat.completion', 'created': 1706193046, 'model': 'gpt-4-1106-vision-preview', 'usage': {'prompt_tokens': 3000, 'completion_tokens': 411, 'total_tokens': 3411}, 'choices': [{'message': {
#     'role': 'assistant', 'content': 'The given image cannot be displayed, but based on the keypoints provided, we can analyze the human pose and infer the bounding boxes.\n\nFrom the keypoints, we have coordinates that are primarily situated in the upper body region, indicating that the man is likely engaging in the fine-motor task of tying a tie which would be categorized as a "Static" pose type since tying a tie is generally a stationary activity. The body orientation is "Frontal" as the keypoints suggest a front-facing stance. The facial direction is not entirely clear from the keypoints, but typically someone tying a tie would be looking downwards to see their hands and the tie, so we would infer a "Looking Downwards" facial direction.\n\nThe object in context, which in this case would be the tie, has a relationship of "On human," specifically around the neck area, within his hands as he would be using them to tie the knot. \n\nSince we are not provided with an image and based on generalized understanding, a reasonable bounding box adjustment for the man could place his upper body within the frame and would likely encompass the area where one would tie a tie. \n\nThus, the updated bounding boxes with adherence to the image coordinate system [0, 0] to [512, 512] could be approximated as follows:\n\n```python\nUpdated Human Location: [50, 210, 280, 390] # Adjusted to frame most of the upper body, head, and hands based on keypoints\nUpdated Object Location: [180, 270, 230, 310] # Adjusted to cover the presumed area of the tie around the neck and hands\n```\n\nThe adjustments are made to ensure the bounding box for the human captures the keypoints related to the upper body and hands without going beyond the original human location significantly, keeping minimal modifications in mind. Meanwhile, the object bounding box is refined to be around the neck and hands area, where the tie would be located. Minimal overlap is maintained by not extending the object box far beyond the hands\' keypoints.'}, 'finish_reason': 'stop', 'index': 0}]}
# api_response_2 = {'id': 'chatcmpl-8lB6uP1wm4BvdE5Mha8K06sCrfqrq', 'object': 'chat.completion', 'created': 1706254816, 'model': 'gpt-4-1106-vision-preview', 'usage': {'prompt_tokens': 3001, 'completion_tokens': 799, 'total_tokens': 3800}, 'choices': [{'message': {
#     'role': 'assistant', 'content': 'To assist the user with the prompt: "A realistic scene of an old woman holding a motorcycle," let\'s analyze the provided keypoints:\n\nKeypoints Analysis:\n- The keypoints provided are primarily clustered at the upper section of the image which might suggest that the person in the image is either standing or seated with an upright stance.\n- Keypoints at the top (285, 84) to (288, 76) suggest the position of the head which is quite high up in the image, indicating the person may be standing.\n- The keypoints around (311, 106), (314, 96), and extending to (286, 156), (289, 144) suggest the line of the upper body and arm positions.\n- The keypoints that extend down to (256, 180) and (261, 169) hint at the lower arm positions, likely where the hands would be holding onto something.\n- The distribution of keypoints from (362, 178) to (347, 297) indicates the side and the lower extremity of the body and feeds into understanding the leg positions, possibly spread apart, indicating a standing pose.\n\nPose Types: "Dynamic" - The act of holding onto a motorcycle suggests the person is not in a relaxed, static pose but rather engaged in a dynamic interaction.\nBody Orientation: "Angled" - As the keypoints suggest a varying distribution of body parts, it is likely the body is not directly frontal or profile but angled.\nFacial Direction: "Directly at Viewer" - Without specific keypoints to suggest the direction the face is looking, the default assumption is directly at the viewer.\nObject Relationship: "Beside human" - The motorcycle would generally be beside the human if it is being held by the person.\n\nThe original bounding box for the human is given as [21, 200, 168, 467], which does not match the keypoints\' suggested location. The bounding box for the motorcycle is given as [151, 104, 446, 348], and as per the keypoints, it should be adjusted to reflect the actual position of the object in relation to the person holding it.\n\nGiven the analysis, the bounding boxes should be adjusted as follows:\n\nUpdated Human Location: Based on the keypoints, an approximate bounding box for the human can be set around the upper body, arms, and head, as it suggests that parts of the body are extending off the current bounding box:\n- Top-left x: Based on the leftmost keypoints at the arms, approximately 249.\n- Top-left y: Based on the topmost keypoint at the head, approximately 74.\n- Bottom-right x: Based on the rightmost keypoints, approximately 362.\n- Bottom-right y: The lowest keypoint is at 305, but considering a full body presence, it would extend further down, approximately to 350 or slightly below.\n\nUpdated Object Location: The motorcycle would need to be beside the woman. Given traditional motorcycle dimensions and proportions, we can assume it extends further both horizontally and downwards:\n- Top-left x: Should be close to the left human keypoints, approximately 249 to stay beside the human.\n- Top-left y: Slightly below the human\'s hand keypoints, approximately 180.\n- Bottom-right x: Assuming the object is quite large and extends beyond the human, around 446 (similar to the original).\n- Bottom-right y: Definitely below the human\'s lower keypoints, expecting around 400 for a realistic proportion.\n\nHowever, the photo provided does not depict an old woman holding a motorcycle but rather a person riding a motorcycle. Therefore, we cannot visually confirm the keypoints, and the adjustments suggested are purely based upon the keypoints provided by the prompt.\n\nHere is the final bounding box information in Python format:\n\n```python\nUpdated_Human_Location = [249, 74, 362, 350]\nUpdated_Object_Location = [249, 180, 446, 400]\n```'}, 'finish_reason': 'stop', 'index': 0}]}
# api_response_3 = {'id': 'chatcmpl-8lZRL1Xx8yBCLXXKdkwfjGsyMM4yj', 'object': 'chat.completion', 'created': 1706348339, 'model': 'gpt-4-1106-vision-preview', 'usage': {'prompt_tokens': 3002, 'completion_tokens': 574, 'total_tokens': 3576}, 'choices': [{'message': {
#     'role': 'assistant', 'content': 'Based on the keypoints provided and the scene described, we have a young man carrying a surfboard. The keypoints indicate the configuration of the human pose, while the image shows us the context of the entire scene.\n\nKeypoints Analysis:\n1. The keypoints for the head (nose through right eye outer) suggest the face is oriented towards the camera, hence we can expect a pose that is angled or profile, with the facial direction being "Directly at Viewer".\n2. The upper body keypoints (shoulders through wrists) indicate that the arms are extended downwards, consistent with carrying an object like a surfboard on the side.\n3. The lower body keypoints (hips through foot index) imply a standing posture, with the body weight distributed evenly across both legs, typical of a "Static" pose type.\n\nPose Types: "Static"\nBody Orientation: "Angled"\nFacial Direction: "Directly at Viewer"\nObject Relationship: "Beside human"\n\nUpdated Human Location:\n- The original human location bounding box [12, 249, 61, 432] does not correspond with the keypoints. We need to adjust it to encapsulate the entire human form indicated by the keypoints.\n- The leftmost point is the left thumb at [206, 289], and the rightmost point could be deduced from the right ankle at [227, 356].\n- The topmost point is within the head keypoints, approximately around [212, 213], and the bottommost point can be considered around the area of the right foot index at [225, 298].\n\nConsidering these keypoints, the bounding box should encapsulate the full height of the young man and the width across his arms, assuming he is carrying the surfboard on his side. In the given keypoint set, the spread of body keypoints does not strongly suggest a wide stance, so we\'ll accommodate a reasonable space for the surfboard being carried.\n\nUpdated Human Location: [200, 213, 235, 360]\n\nUpdated Object Location:\n- The original object location bounding box [263, 368, 375, 481] needs to be adjusted to reflect the position of the surfboard next to the young man.\n- Since the surfboard is carried beside the human, the box must be adjusted to be next to the updated human bounding box, maintaining no gap as the surfboard would be in contact with the person.\n\nWe cannot see the entire length of the surfboard from the keypoints, but assuming a regular surfboard size, it would likely extend beyond the human\'s head and reach towards the lower legs.\n\nUpdated Object Location: [236, 170, 300, 360]\n\nThe updated bounding boxes reflect the position of the human and the object (surfboard) beside him, matching the keypoints provided and maintaining visual accuracy according to the guide image.'}, 'finish_reason': 'stop', 'index': 0}]}
# api_response_4 = {'id': 'chatcmpl-8laAnL8ZaD3cOxxf11MvImUhD3jbB', 'object': 'chat.completion', 'created': 1706351157, 'model': 'gpt-4-1106-vision-preview', 'usage': {'prompt_tokens': 3000, 'completion_tokens': 471, 'total_tokens': 3471}, 'choices': [{'message': {'role': 'assistant', 'content': 'The provided keypoints correspond to a dynamic pose of a man riding on a skateboard. The keypoints suggest movements associated with balancing and motion typically seen when skateboarding, such as extended arms and legs positioned to maneuver the board. The specific arrangement of keypoints indicates that the man\'s body is angled away from the viewer, with the arms and one leg extended outward to maintain balance.\n\nBased on the keypoints provided, we can discern the following about the pose and situation:\n\n- Pose Type: "Dynamic," as the activity of riding a skateboard implies motion and balance.\n- Body Orientation: "Angled," given the asymmetric arrangement of the keypoints across the body.\n- Facial Direction: "Looking Sideways," assuming the person\'s face is oriented towards the same direction as the shoulder and hip keypoints on one side of the body.\n- Object Relationship: "Under human," as the skateboard is a device that is typically ridden on top of.\n  \nGiven the provided keypoints and the described content, I will adjust the bounding boxes according to this analysis:\n\n- Updated Human Location: Considering the pose and that no complete information on the height of the man is given, conservatively updating the bounding box to be centered on the balance point suggested by the keypoints while ensuring it encompasses all keypoints would yield approximately [210, 40, 430, 365]. This estimation takes into account the observed dynamic nature and the spatial distribution of the keypoints.\n  \n- Updated Object Location: Since the object in question is a skateboard that would be located under the man, I will update the bounding box location to fit directly under the man\'s center of gravity and extended leg. Assuming the skateboard is not entirely visible and taking the dynamic position into account, this would lead to the bounding box being approximately [240, 285, 330, 390].\n\nPlease note that these are approximations given the constraints of estimating from keypoints without a reference image to clarify exact proportions and positions. The actual values could vary based on the specific scene and object placements. \n\nAs for the image provided, I cannot identify the person in the image or discuss any personal characteristics, but I can note that it appears to involve the dynamic activity of skateboarding, which aligns with the user prompt. However, I cannot provide bounding box updates with reference to an actual image as it requires an analysis purely based on keypoints and written descriptions.'}, 'finish_reason': 'stop', 'index': 0}]}

# print("Test Case 1:")
# human_location_1, object_location_1 = parse_bboxes(api_response_1)

# print("\nTest Case 2:")
# human_location_2, object_location_2 = parse_bboxes(api_response_2)

# print("\nTest Case 3:")
# human_location_3, object_location_3 = parse_bboxes(api_response_3)

# print("\nTest Case 4:")
# human_location_4, object_location_4 = parse_bboxes(api_response_4)
