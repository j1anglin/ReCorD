import json
import numpy as np


def read_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)


# Function to format bounding boxes as strings
def format_bboxes(bboxes):
    return '[' + '], ['.join(', '.join(map(str, bbox)) for bbox in bboxes) + ']'


def bbox_str_to_list(bbox_str):
    """Converts a bounding box string to a list of integers."""
    try:
        return [list(map(int, bbox.split(', '))) for bbox in bbox_str.strip('[]').split('], [') if bbox]
    except ValueError as e:
        print(f"Error converting bbox string to list: {e}")
        print(f"BBox string: {bbox_str}")
        return []


def bbox_area(bbox):
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def analyze_bbox_sizes(annotations):
    human_bbox_sizes = []
    object_bbox_sizes = []

    for entry in annotations:
        for hoi in entry['hois']:
            human_bboxes = hoi['human_bboxes']
            object_bboxes = hoi['object_bboxes']

            for bbox in human_bboxes:
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                area = width * height
                human_bbox_sizes.append(area)

            for bbox in object_bboxes:
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                area = width * height
                object_bbox_sizes.append(area)

    # Compute statistics
    human_stats = {
        'mean': np.mean(human_bbox_sizes),
        'std_dev': np.std(human_bbox_sizes),
        'min': np.min(human_bbox_sizes),
        'max': np.max(human_bbox_sizes),
        'percentiles': {
            '25th': np.percentile(human_bbox_sizes, 25),
            '50th': np.percentile(human_bbox_sizes, 50),
            '75th': np.percentile(human_bbox_sizes, 75),
        }
    }

    object_stats = {
        'mean': np.mean(object_bbox_sizes),
        'std_dev': np.std(object_bbox_sizes),
        'min': np.min(object_bbox_sizes),
        'max': np.max(object_bbox_sizes),
        'percentiles': {
            '25th': np.percentile(object_bbox_sizes, 25),
            '50th': np.percentile(object_bbox_sizes, 50),
            '75th': np.percentile(object_bbox_sizes, 75),
        }
    }

    return human_stats, object_stats


def analyze_counts(annotations):
    human_counts = []
    object_counts = []

    for entry in annotations:
        for hoi in entry['hois']:
            human_counts.append(len(hoi['human_bboxes']))
            object_counts.append(len(hoi['object_bboxes']))

    # Convert to NumPy arrays for efficient computation
    human_counts = np.array(human_counts)
    object_counts = np.array(object_counts)

    # Compute statistics including percentiles
    human_stats = {
        'mean': np.mean(human_counts),
        'std_dev': np.std(human_counts),
        'min': np.min(human_counts),
        'max': np.max(human_counts)
    }

    object_stats = {
        'mean': np.mean(object_counts),
        'std_dev': np.std(object_counts),
        'min': np.min(object_counts),
        'max': np.max(object_counts)
    }

    return human_stats, object_stats


def filter_annotations(annotations, min_bbox_area, max_human_bboxes):
    filtered_annotations = []
    for annotation in annotations:
        human_bboxes = bbox_str_to_list(annotation['human_bboxes'])
        object_bboxes = bbox_str_to_list(annotation['object_bboxes'])

        if len(human_bboxes) <= max_human_bboxes:
            filtered_human_bboxes = [
                bbox for bbox in human_bboxes if bbox_area(bbox) >= min_bbox_area]
            filtered_object_bboxes = [
                bbox for bbox in object_bboxes if bbox_area(bbox) >= min_bbox_area]

            if filtered_human_bboxes and filtered_object_bboxes:
                annotation['human_bboxes'] = format_bboxes(
                    filtered_human_bboxes)
                annotation['object_bboxes'] = format_bboxes(
                    filtered_object_bboxes)
                filtered_annotations.append(annotation)
    return filtered_annotations


def match_and_filter_annotations(annotations_file, objects_verbs_file, min_bbox_area, max_human_bboxes):
    annotations = read_json(annotations_file)
    objects_verbs = read_json(objects_verbs_file)

    # Creating a dictionary for easy access to object and verb by id
    id_to_object_verb = {str(item['id']): {
        'object': item['object'], 'verb': item['verb']} for item in objects_verbs}

    # Prepare the matched_annotations list
    matched_annotations = []
    for entry in annotations:
        global_id = entry['global_id']
        for hoi in entry['hois']:
            hoi_id = str(hoi['id'])
            if hoi_id in id_to_object_verb:
                matched_annotation = {
                    'global_id': global_id,
                    'object': id_to_object_verb[hoi_id]['object'],
                    'verb': id_to_object_verb[hoi_id]['verb'],
                    'human_bboxes': format_bboxes(hoi['human_bboxes']),
                    'object_bboxes': format_bboxes(hoi['object_bboxes']),
                    # 'human_bboxes_shifted': format_bboxes(hoi['human_bboxes_shifted']),
                    'object_bboxes_shifted': format_bboxes(hoi['object_bboxes_shifted'])
                }
                matched_annotations.append(matched_annotation)
            else:
                print(f"No match found for HOI ID: {hoi_id}")

    return filter_annotations(matched_annotations, min_bbox_area, max_human_bboxes)


# Paths to your JSON files
annotations_file = "hico_20160224_det_processed/anno_list_512_shifted.json"
objects_verbs_file = "hico_20160224_det_processed/hoi_list.json"

# Use the raw_annotations for the analysis
# raw_annotations = read_json(annotations_file)
# human_stats, object_stats = analyze_bbox_sizes(raw_annotations)
# human_count_stats, object_count_stats = analyze_counts(raw_annotations)

# print("Human Bounding Box Statistics:", human_stats)
# print("Object Bounding Box Statistics:", object_stats)
# print("Human Count Statistics:", human_count_stats)
# print("Object Count Statistics:", object_count_stats)

print("Human Bounding Box Statistics: {'mean': 47307.499145741174, 'std_dev': 57556.42735416371, 'min': 5, 'max': 260100, 'percentiles': {'25th': 5060.0, '50th': 24396.0, '75th': 67124.0}}")
print("Object Bounding Box Statistics: {'mean': 43809.052235052266, 'std_dev': 54980.08968470133, 'min': 9, 'max': 260610, 'percentiles': {'25th': 4192.0, '50th': 18321.0, '75th': 66760.75}}")
print(
    "Human Count Statistics: {'mean': 1.5386725022380743, 'std_dev': 1.8294325336358785, 'min': 0, 'max': 57}")
print(
    "Object Count Statistics: {'mean': 1.2995324911542778, 'std_dev': 1.456189804721253, 'min': 0, 'max': 106}")

# Minimum bounding box area in pixels
min_bbox_area = 5000
print("\nmin_bbox_area:", min_bbox_area)

# Maximum number of human bounding boxes
max_human_bboxes = 1
print("max_human_bboxes:", max_human_bboxes)

# Generate the matched_annotations list
matched_annotations = match_and_filter_annotations(
    annotations_file, objects_verbs_file, min_bbox_area, max_human_bboxes)

# Save the matched_annotations to a JSON file
with open('./GPT4V/matched_annotations_filtered.json', 'w') as file:
    json.dump(matched_annotations, file, indent=4)
