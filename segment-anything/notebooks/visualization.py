import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import sys
import cv2
import os
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from PIL import Image

def get_model():
    sam_checkpoint = "../sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(sam)
    return mask_generator

def get_anns_score(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['predicted_iou']), reverse=True)
    ann_avg =  sorted_anns[0]['segmentation'].astype(np.float32)
    return ann_avg
    
def generate_slice(image_data, slice_index):
    mask_generator = get_model()
    # Extract the i-th slice (2D array)
    slice_data = image_data[:, :, slice_index]
    # Normalize the slice to 0-255 range
    slice_normalized = cv2.normalize(slice_data, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    slice_normalized = slice_normalized.astype(np.uint8)
    pseudo_rgb = cv2.cvtColor(slice_normalized, cv2.COLOR_GRAY2RGB)
    mask = mask_generator.generate(pseudo_rgb)
    final_mask = get_anns_score(mask)
    return final_mask
     
def visualize_slice_with_sam_mask(image_data, true_mask_data, sam_mask_slice, slice_index, save_path=None):
    """
    Visualize a single 2D slice from 3D image data with its true mask and SAM-generated mask.

    Args:
    - image_data (numpy.ndarray): The 3D image data.
    - true_mask_data (numpy.ndarray): The 3D true mask data.
    - sam_mask_data (numpy.ndarray): The 3D SAM-generated mask data.
    - slice_index (int): The index of the slice to visualize.
    - save_path (str, optional): Path to save the visualization. If None, the image will not be saved.

    Returns:
    - matplotlib figure and axes objects
    """

    if slice_index < 0 or slice_index >= image_data.shape[2]:
        raise ValueError("slice_index out of range")

    # Get the specific slice from image, true mask, and SAM mask
    image_slice = image_data[:, :, slice_index]
    true_mask_slice = true_mask_data[:, :, slice_index]
    #sam_mask_slice = sam_mask_data[:, :, slice_index]

    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    # Display the image slice
    ax[0].imshow(image_slice, cmap='gray')
    ax[0].set_title(f'Image Slice {slice_index}')
    ax[0].axis('off')

    # Display the image slice with true mask overlay
    #ax[1].imshow(image_slice, cmap='gray')
    ax[1].imshow(true_mask_slice, cmap='gray')
    #ax[1].imshow(true_mask_slice, cmap='jet', alpha=0.5)  # Adjust alpha for mask transparency
    print(true_mask_slice)
    ax[1].set_title(f'True Mask Overlay')
    ax[1].axis('off')

    # Display the image slice with SAM mask overlay
    # ax[2].imshow(image_slice, cmap='gray')
    # ax[2].imshow(sam_mask_slice, cmap='jet', alpha=0.5)
    ax[2].imshow(sam_mask_slice, cmap='gray')
    ax[2].set_title(f'SAM Mask Overlay')
    ax[2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    return fig, ax


# Example usage, process data
slice_index = 50  # Choose the index of the slice to visualize
image = nib.load('/home/jiaxin/ML23Fall/data/images/RawData/Training/img/img0001.nii.gz')
label_image = nib.load('/home/jiaxin/ML23Fall/data/images/RawData/Training/label/label0001.nii.gz')
image_data = image.get_fdata()
label_data = label_image.get_fdata()
sam_generated_masks = generate_slice(image_data, slice_index)
visualize_slice_with_sam_mask(image_data, label_data, sam_generated_masks, slice_index, save_path='./images/image_visualization.png')