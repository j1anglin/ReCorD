import os
import ast
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from PIL import Image, ImageEnhance


def plot_bounding_boxes(image_path, updated_object_location, prompt, human_bbox, object_bbox, human_bboxes_shifted=None, object_bboxes_shifted=None):
    def parse_bbox_string(bbox_string):
        """Parse a bounding box string into a tuple."""
        try:
            return tuple(ast.literal_eval(bbox_string))
        except Exception as e:
            print(f"Error parsing bounding box string: {e}")
            return None

    def plot_bbox(ax, bbox, color, label):
        """Plot a single bounding box."""
        if bbox:
            rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                                     linewidth=1, edgecolor=color, facecolor='none', label=label)
            ax.add_patch(rect)

    # Convert string representations to tuples
    # human_bbox = parse_bbox_string(human_bbox_str)
    # object_bbox = parse_bbox_string(object_bbox_str)
    # human_bboxes_shifted = parse_bbox_string(human_bboxes_shifted_str)
    # object_bboxes_shifted = parse_bbox_string(object_bboxes_shifted_str)

    # Load and resize the image
    image = Image.open(image_path)

    # Enhance the image (dim the brightness)
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(0.2)  # Adjust the factor to control the brightness

    image = image.resize((512, 512))
    fig, ax = plt.subplots()

    # Display the image
    ax.imshow(image)

    # Plot original and updated bounding boxes
    plot_bbox(ax, human_bbox, 'b', 'Original Human')
    plot_bbox(ax, object_bbox, 'g', 'Original Object')
    plot_bbox(ax, human_bboxes_shifted, 'c', 'Shifted Human')
    plot_bbox(ax, object_bboxes_shifted, 'm', 'Shifted Object')
    # plot_bbox(ax, updated_human_location, 'r', 'Updated Human')
    plot_bbox(ax, updated_object_location, 'y', 'Updated Object')

    # Add legend outside the figure
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    # Add title
    plt.title(prompt)

    # Save the figure
    output_directory = 'GPT4V/examples'
    # Create directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    output_file = os.path.join(output_directory, prompt)
    plt.savefig(output_file, bbox_inches='tight')
    print(f"Saved figure to {output_file}")

    plt.show()

# Example usage
# train_image_path = "./hico_20160224_det/images/train2015/"
# image_name = "HICO_train2015_00013461"
# image_path = f"{train_image_path}{image_name}.jpg"
# updated_human_location = (100, 50, 350, 250)
# updated_object_location = (100, 150, 400, 300)
# prompt = "A teenager is riding a bicycle"
# human_bbox = "[133, 26, 371, 247]"
# object_bbox = "[144, 140, 432, 289]"
# human_bboxes_shifted = [178, 234, 416, 455]
# object_bboxes_shifted = [142, 293, 430, 442]

# plot_bounding_boxes(image_path, updated_human_location, updated_object_location,
#                     prompt, human_bbox, object_bbox, human_bboxes_shifted, object_bboxes_shifted)
