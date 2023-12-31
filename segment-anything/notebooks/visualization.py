import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
def visualize_slice_with_mask(image_data, mask_data, slice_index, save_path=None):
    """
    Visualize a single 2D slice from 3D image data with its corresponding mask.

    Args:
    - image_data (numpy.ndarray): The 3D image data.
    - mask_data (numpy.ndarray): The 3D mask data.
    - slice_index (int): The index of the slice to visualize.
    - save_path (str, optional): Path to save the visualization. If None, the image will not be saved.

    Returns:
    - matplotlib figure and axes objects
    """

    if slice_index < 0 or slice_index >= image_data.shape[2]:
        raise ValueError("slice_index out of range")

    # Get the specific slice from both image and mask
    image_slice = image_data[:, :, slice_index]
    mask_slice = mask_data[:, :, slice_index]

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Display the image slice
    ax[0].imshow(image_slice, cmap='gray')
    ax[0].set_title(f'Image Slice {slice_index}')
    ax[0].axis('off')

    # Display the image slice with mask overlay
    ax[1].imshow(image_slice, cmap='gray')
    ax[1].imshow(mask_slice, cmap='jet', alpha=0.5)  # Adjust alpha for mask transparency
    ax[1].set_title(f'Image Slice {slice_index} with Mask')
    ax[1].axis('off')

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

visualize_slice_with_mask(image_data, label_data, slice_index, save_path='./images/image_visualization.png')
