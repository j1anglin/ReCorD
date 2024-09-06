import torch
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
import os
import datetime
import math
import numbers
import torch
from torch import nn
from torch.nn import functional as F


def create_otsu_mask(img_tensor):
    """
    Calculate Otsu's threshold for a given grayscale image tensor.

    Args:
    img_tensor (torch.Tensor): A grayscale image tensor.

    Returns:
    float: The Otsu's threshold.
    """
    # Flatten the image
    pixels = img_tensor.flatten().int()
    
    # Calculate histogram
    hist = torch.histc(pixels.float(), bins=256, min=0, max=255)

    # Total number of pixels
    total = pixels.shape[0]

    current_max, threshold = 0, 0
    sum_total, sum_foreground, weight_background = 0, 0, 0

    for i in range(256):
        sum_total += i * hist[i]

    for i in range(256):
        weight_background += hist[i]
        if weight_background == 0:
            continue

        weight_foreground = total - weight_background
        if weight_foreground == 0:
            break

        sum_foreground += i * hist[i]

        mean_background = sum_foreground / weight_background
        mean_foreground = (sum_total - sum_foreground) / weight_foreground

        # Calculate Between Class Variance
        var_between = weight_background * weight_foreground
        var_between *= (mean_background - mean_foreground) ** 2

        # Check if new maximum found
        if var_between > current_max:
            current_max = var_between
            threshold = i

            
    mask = img_tensor >= threshold
    mask = (mask * 255).to(torch.uint8)       
    
    
    return mask     



def find_largest_contour_center_coor(mask):
    """
    Find the bounding box (x, y, width, height) of the largest contour in the mask.

    Args:
    mask (numpy.ndarray): A binary mask.

    Returns:
    tuple: A tuple (x, y, w, h) representing the bounding box of the largest contour.
    """
    # Finding contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Finding the largest contour based on area
    largest_area = 0
    largest_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > largest_area:
            largest_area = area
            largest_contour = contour

    if largest_contour is not None:
        # Calculating the bounding box of the largest contour
        x, y, w, h = cv2.boundingRect(largest_contour)
        return x, y, w, h
    else:
        return None

def process_and_save_images(img_tensors, center_coors, save_dir='./'):
    """
    Process the given list of grayscale image tensors, create Otsu masks,
    find the largest contour for each, and save the masks with bounding boxes to files.
    
    Args:
    img_tensors (list of torch.Tensor): A list of grayscale image tensors.
    center_coors (list of tuple): List of tuples for the largest contour center coordinates for each image.
    save_dir (str): Directory path to save the images.
    """

    for i, img_tensor in enumerate(img_tensors):
        # Ensure the directory exists
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Convert tensor to numpy for cv2 processing
        #otsu_mask_numpy = img_tensor.numpy()  # Assuming the tensor is already in a format compatible with cv2

        # Find largest contour center coordinates
        bbox = center_coors[i]

        # Convert to PIL image for saving
        img = Image.fromarray(img_tensor)

        # Draw bounding box if it exists
        if bbox:
            x, y, w, h = bbox
            print(f'(x, y, w, h): ({x}, {y}, {w}, {h})')
            draw = ImageDraw.Draw(img)
            draw.rectangle([x, y, x + w, y + h], outline='red', width=1)

        # Resize the image to 256x256 pixels
        img = img.resize((256, 256))
        timestamp = datetime.datetime.now().strftime("%d%H%M%S")
        # Save the image with a unique filename for each tensor
        save_path = os.path.join(save_dir, f'otsu_mask_{timestamp}.jpg')
        img.save(save_path)
        print(f'Image {i} saved to {save_path}')

        


def get_operations_and_modify_indices(indices_to_alter, attributes_belongings, source_coors, target_coors):
    

    modify_indices = []
    operations = []
    for i, index in enumerate(indices_to_alter):
        """
        if(attributes_belongings[i]):
            modify_indices.append([index] + attributes_belongings[i])
        else:
            modify_indices.append([index])
        """
        modify_indices.append([index])

    for i, source_coor in enumerate(source_coors):
        if(source_coor==target_coors[i]):
            operations.append(["None"])

        if (source_coor[:2]!=target_coors[i][:2] and source_coor[-2:]!=target_coors[i][-2:]):
            operations.append(["Resize", "Shift"])  
        elif(source_coor[-2:]!=target_coors[i][-2:]):
            operations.append(["Resize"])        
        elif(source_coor[:2]!=target_coors[i][:2]):    
            operations.append(["Shift"])

    return operations, modify_indices


# +



class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight.to(input.dtype), groups=self.groups)


class AverageSmoothing(nn.Module):
    """
    Apply average smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the average kernel.
        sigma (float, sequence): Standard deviation of the rage kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, dim=2):
        super(AverageSmoothing, self).__init__()

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = torch.ones(size=(kernel_size, kernel_size)) / (kernel_size * kernel_size)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply average filter to input.
        Arguments:
            input (torch.Tensor): Input to apply average filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)
