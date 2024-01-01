import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import sys
import cv2
import os
import torch
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from PIL import Image
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
)

from monai.config import print_config
from monai.metrics import DiceMetric
from monai.networks.nets import UNETR

from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
)
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-175,
            a_max=250,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(96, 96, 96),
            pos=1,
            neg=1,
            num_samples=4,
            image_key="image",
            image_threshold=0,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[0],
            prob=0.10,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[1],
            prob=0.10,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[2],
            prob=0.10,
        ),
        RandRotate90d(
            keys=["image", "label"],
            prob=0.10,
            max_k=3,
        ),
        RandShiftIntensityd(
            keys=["image"],
            offsets=0.10,
            prob=0.50,
        ),
    ]
)

def get_model():
    sam_checkpoint = "../sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(sam)
    return mask_generator, sam

def get_anns_score(anns):
    if len(anns) == 0:
        return
    #sorted_anns = sorted(anns, key=(lambda x: x['predicted_iou']), reverse=True)
    sorted_anns = sorted(anns, key=(lambda x: x['predicted_iou']), reverse=True)
    sorted_anns = sorted_anns[:13]
    ann_avg =  sorted_anns[0]['segmentation'].astype(np.float32)
    for i, ann in enumerate(sorted_anns[1:]):
        ann_avg +=  (i+1) * ann['segmentation'].astype(np.float32)
    return ann_avg
    
def generate_slice(image_data, slice_index):
    mask_generator, sam = get_model()
    print(image_data)
    # Extract the i-th slice (2D array)
    slice_data = image_data[0, :, :, slice_index].numpy()
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

def main():
    # Example usage, process data
    slice_index = 50  # Choose the index of the slice to visualize
    image = nib.load('/home/jiaxin/ML23Fall/data/images/RawData/Training/img/img0002.nii.gz')
    label_image = nib.load('/home/jiaxin/ML23Fall/data/images/RawData/Training/label/label0002.nii.gz')
    print(label_image)
    image_data = image.get_fdata()
    label_data = label_image.get_fdata()
    print("Label data shape:", label_data.shape)
    print("Unique values in label data slice:", np.unique(label_data))

    sam_generated_masks = generate_slice(image_data, slice_index)
    visualize_slice_with_sam_mask(image_data, label_data, sam_generated_masks, slice_index, save_path='./images/image_visualization.png')
    
val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
    ]
)
data_dir = "/home/jiaxin/ML23Fall/data/images/RawData/Training/"
split_json = "dataset_0.json"

datasets = data_dir + split_json
datalist = load_decathlon_datalist(datasets, True, "training")
val_files = load_decathlon_datalist(datasets, True, "validation")
train_ds = CacheDataset(
    data=datalist,
    transform=train_transforms,
    cache_num=24,
    cache_rate=1.0,
    num_workers=8,
)
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_num=6, cache_rate=1.0, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
slice_map = {
    "img0035.nii.gz": 170,
    "img0036.nii.gz": 230,
    "img0037.nii.gz": 204,
    "img0038.nii.gz": 204,
    "img0039.nii.gz": 204,
    "img0040.nii.gz": 180,
}
case_num = 0
img_name = os.path.split(val_ds[case_num]["image"].meta["filename_or_obj"])[1]
img = val_ds[case_num]["image"]
label = val_ds[case_num]["label"]
img_shape = img.shape
label_shape = label.shape
print(f"image shape: {img_shape}, label shape: {label_shape}")
plt.figure("image", (18, 6))
plt.subplot(1, 3, 1)
plt.title("image")
print(img)
plt.imshow(img[0, :, :, slice_map[img_name]].detach().cpu(), cmap="gray")
plt.subplot(1, 3, 2)
plt.title("label")
plt.imshow(label[0, :, :, slice_map[img_name]].detach().cpu())
sam_generated_masks = torch.tensor(generate_slice(img, slice_map[img_name]))
plt.subplot(1, 3, 3)
plt.title("sam mask")
plt.imshow(sam_generated_masks.detach().cpu())
plt.show()
output_path = './images/image.png'  # Change to your desired path
plt.savefig(output_path)
