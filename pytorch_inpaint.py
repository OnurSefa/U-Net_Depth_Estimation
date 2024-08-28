import torch
from torchvision.transforms.functional import pil_to_tensor
import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2


def fast_fill_depth_map(depth_map):
    # Ensure the depth map is in 32-bit float format
    depth_map = depth_map.astype(np.float32)

    # Normalize the depth map to range [0, 1]
    depth_map = (depth_map - np.nanmin(depth_map)) / (np.nanmax(depth_map) - np.nanmin(depth_map))

    # Create a mask for missing values (assuming 0 or NaN represents missing data)
    mask = np.zeros(depth_map.shape, dtype=np.uint8)
    mask[np.isnan(depth_map) | (depth_map == 0)] = 255

    # Replace NaN values with zeros to make the array valid for OpenCV
    depth_map[np.isnan(depth_map)] = 0

    # Perform inpainting using OpenCV (INPAINT_NS or INPAINT_TELEA)
    filled_depth = cv2.inpaint(depth_map, mask, 3, cv2.INPAINT_NS)

    return filled_depth


def guided_filter_inpainting(depth_map, rgb_image):

    rgb_image = np.transpose(rgb_image, (1, 2, 0))
    # Apply the guided filter with the RGB image as guidance
    guided_depth = cv2.ximgproc.guidedFilter(
        guide=rgb_image.astype(np.float32),
        src=depth_map.astype(np.float32),
        radius=9,  # Filter radius
        eps=1  # Regularization term, depends on the range of the values
    )

    return guided_depth


def inpainting_all(dp_path, out_dir):
    tensor_to_pil = T.ToPILImage(mode='L')  # Use mode 'F' for floating point images

    with open(dp_path, 'r') as f:
        data_points = json.load(f)

    for image_path, depth_path in data_points:
        image = Image.open(image_path)
        depth = Image.open(depth_path)

        # Convert images to tensor and float
        image = pil_to_tensor(image).float()
        depth = pil_to_tensor(depth).float()

        # Convert tensors to numpy arrays
        image = image.squeeze().numpy()  # Remove extra channel dimension if it exists
        depth = depth.squeeze().numpy()  # Remove extra channel dimension if it exists

        # Fill the depth map
        depth = fast_fill_depth_map(depth)

        # Apply guided filter
        # depth = guided_filter_inpainting(depth, image)

        # Convert the filled depth map back to a torch tensor and then to a PIL image
        depth = torch.from_numpy(depth)
        depth = tensor_to_pil(depth)
        depth.save('image_inpainting.png')

        print('Inpainting with guided filtering completed and saved as image.png')
