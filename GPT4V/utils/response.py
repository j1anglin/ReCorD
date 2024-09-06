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
