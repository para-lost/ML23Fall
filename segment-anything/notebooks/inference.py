import nibabel as nib
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
import os
import random
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from PIL import Image

import nibabel as nib
import numpy as np
from scipy import ndimage


val_img_names = [
    "img0035.nii.gz",
    "img0036.nii.gz",
    "img0037.nii.gz",
    "img0038.nii.gz",
    "img0039.nii.gz",
    "img0040.nii.gz",
]
val_label_names = [
    "label0035.nii.gz",
    "label0036.nii.gz",
    "label0037.nii.gz",
    "label0038.nii.gz",
    "label0039.nii.gz",
    "label0040.nii.gz",
]
def get_image(image_data, slice):
    slice_data = image_data[:, :, slice]
    # Normalize the slice to 0-255 range
    slice_normalized = cv2.normalize(slice_data, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # Convert to uint8
    slice_normalized = slice_normalized.astype(np.uint8)

    # Convert grayscale to pseudo-RGB by replicating the channel
    pseudo_rgb = cv2.cvtColor(slice_normalized, cv2.COLOR_GRAY2RGB)
    # pseudo_rgb = slice_normalized
    return pseudo_rgb

def get_predictor():
    sam_checkpoint = "../sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)
    return predictor
predictor = get_predictor()
def sam_generate_mask(image, input_point):
    predictor.set_image(image)
    input_label = np.array([1])
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )
    return masks[0]
    
# Function to extract prompts from a single slice
def extract_multiple_prompts_from_slice(slice_data, class_value):
    prompts = []
    mask = slice_data == class_value
    if np.any(mask):
        # Example: find the center of the mask
        center_of_mass = ndimage.measurements.center_of_mass(mask)
        prompts.append(center_of_mass)
        # Add more prompt types as needed
    return prompts

# Function to extract prompts from a single slice
def extract_single_prompts_from_slice(slice_data, class_value):
    prompts = []
    mask = slice_data == class_value
    if np.any(mask):
        # Example: find the center of the mask
        center_of_mass = ndimage.measurements.center_of_mass(mask)
        prompts.append(center_of_mass)
        # Add more prompt types as needed
    return prompts

def find_geometric_center(mask):
    ans_list = []
    ans = list(ndimage.measurements.center_of_mass(mask))
    ans_list.append(ans)
    ans_list = np.array(ans_list)
    return ans_list

def find_random_point(mask):
    points = np.argwhere(mask)
    ans = list(points[random.randint(0, len(points) - 1)])
    return np.array([ans])
    

def find_multiple_points(mask, num_points=5):
    points = [find_geometric_center(mask)[0]]
    while len(points) < num_points:
        point = find_random_point(mask)
        points.append(point)
    return np.array(points)

def find_bounding_box(mask):
    slices = ndimage.find_objects(mask)
    return slices[0] if slices else None

def mdice(map1, map2):
    map1 = torch.tensor(map1)
    map2 = torch.tensor(map2)
    return 2*torch.sum(map1 == map2)/(torch.sum(map1 == map1)+torch.sum(map2 == map2))

def inference_for_class(label_data, image_data, class_value):
    single_val = 0
    multi_val = 0
    bbox_val = 0
    single_num = 0
    multi_num = 0
    bbox_num = 0
    for slice_index in range(label_data.shape[2]):
        slice_data = label_data[:, :, slice_index]
        slice_image = get_image(image_data, slice_index)
        mask = slice_data == class_value
        if np.any(mask):
            # Single point prompt
            print(find_random_point(mask))
            sam_mask_random = sam_generate_mask(slice_image, find_random_point(mask))
            single_val += mdice(sam_mask_random, mask)
            single_num += 1
            print(single_val)
            # Multiple points prompt
            print(find_multiple_points(mask))

            # Bounding box prompt
            bbox = find_bounding_box(mask)
            if bbox:
                print(bbox)

    return single_val/single_num, 0, 0

def visualization(image_data, label_data, class_value):
    for slice_index in range(label_data.shape[2]):
        slice_data = label_data[:, :, slice_index]
        slice_image = get_image(image_data, slice_index)
        mask = slice_data == class_value
        if np.any(mask):
            # Single point prompt
            print(find_random_point(mask))
            sam_mask_random = sam_generate_mask(slice_image, find_random_point(mask))
            plt.figure("image", (18, 6))
            plt.subplot(1, 3, 1)
            plt.title("image")
            plt.imshow(image_data[:, :, slice_index], cmap="gray")
            plt.subplot(1, 3, 2)
            plt.title("label")
            plt.imshow(mask)
            plt.subplot(1, 3, 3)
            plt.title("sam mask")
            plt.imshow(sam_mask_random)
            plt.show()
            output_path = './images/image2.png'  # Change to your desired path
            plt.savefig(output_path)
            return


def main():
    tot_dict_random = {}
    tot_num = {}
    for i in range(1, 14):
        tot_dict_random[i] = 0
        tot_num[i] = 0
    for img_name, label_name in zip(val_img_names, val_label_names):
        print(img_name)
        image = nib.load('/home/jiaxin/ML23Fall/data/images/RawData/Training/imagesTr/' + img_name)
        label_image = nib.load('/home/jiaxin/ML23Fall/data/images/RawData/Training/labelsTr/' +label_name)
        image_data = image.get_fdata()
        label_data = label_image.get_fdata()
        # Get unique classes (excluding background)
        classes = np.unique(label_data)[1:]  # Excludes 0 if it represents the background
        for class_idx in classes:
            random_score, multi_score, bbox_score = inference_for_class(label_data, image_data, class_idx)
            tot_num[class_idx] += 1
            tot_dict_random[class_idx] += random_score
    for key in tot_dict_random.keys():
        print(key, tot_dict_random[key]/tot_num[key])

def try_visualization():
    img_name = val_img_names[0]
    label_name = val_label_names[0]
    image = nib.load('/home/jiaxin/ML23Fall/data/images/RawData/Training/imagesTr/' + img_name)
    label_image = nib.load('/home/jiaxin/ML23Fall/data/images/RawData/Training/labelsTr/' +label_name)
    image_data = image.get_fdata()
    label_data = label_image.get_fdata()
    # Get unique classes (excluding background)
    classes = np.unique(label_data)[1:]  # Excludes 0 if it represents the background
    for class_idx in classes:
        visualization(image_data, label_data,  class_idx)
        return

try_visualization()