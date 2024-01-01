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
from segment_anything.utils.transforms import ResizeLongestSide
from torch.optim import Adam
import monai
from tqdm import tqdm
from statistics import mean
import torch
from torch.nn.functional import threshold, normalize

from PIL import Image

import nibabel as nib
import numpy as np
from scipy import ndimage
import logging
import time

device="cuda"
val_img_names = [
    "img0001.nii.gz",
    "img0002.nii.gz",
    "img0003.nii.gz",
    "img0004.nii.gz",
    "img0005.nii.gz",
    "img0006.nii.gz",
    "img0007.nii.gz",
    "img0008.nii.gz",
    "img0009.nii.gz",
    "img0010.nii.gz",
    "img0011.nii.gz",
    "img0012.nii.gz",
    "img0013.nii.gz",
    "img0014.nii.gz",
    "img0015.nii.gz",
    "img0016.nii.gz",
    "img0017.nii.gz",
    "img0018.nii.gz",
    "img0019.nii.gz",
    "img0020.nii.gz",
    "img0021.nii.gz",
    "img0022.nii.gz",
    "img0023.nii.gz",
    "img0024.nii.gz",
    "img0025.nii.gz",
    "img0026.nii.gz",
    "img0027.nii.gz",
    "img0028.nii.gz",
    "img0029.nii.gz",
    "img0030.nii.gz",
    "img0031.nii.gz",
    "img0032.nii.gz",
    "img0033.nii.gz",
    "img0034.nii.gz",
]
val_label_names = [
    "label0001.nii.gz",
    "label0002.nii.gz",
    "label0003.nii.gz",
    "label0004.nii.gz",
    "label0005.nii.gz",
    "label0006.nii.gz",
    "label0007.nii.gz",
    "label0008.nii.gz",
    "label0009.nii.gz",
    "label0010.nii.gz",
    "label0011.nii.gz",
    "label0012.nii.gz",
    "label0013.nii.gz",
    "label0014.nii.gz",
    "label0015.nii.gz",
    "label0016.nii.gz",
    "label0017.nii.gz",
    "label0018.nii.gz",
    "label0019.nii.gz",
    "label0020.nii.gz",
    "label0021.nii.gz",
    "label0022.nii.gz",
    "label0023.nii.gz",
    "label0024.nii.gz",
    "label0025.nii.gz",
    "label0026.nii.gz",
    "label0027.nii.gz",
    "label0028.nii.gz",
    "label0029.nii.gz",
    "label0030.nii.gz",
    "label0031.nii.gz",
    "label0032.nii.gz",
    "label0033.nii.gz",
    "label0034.nii.gz",
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

    return sam
    
sam_model = get_predictor()
def sam_generate_mask(image, input_point, input_box = None, input_label = np.array([1])):
    predictor.set_image(image)
    # input_label = np.array([1])
    if input_box is not None:
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            box=input_box[None, :],
            multimask_output=True,
        )
    else:
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            #box=input_box[None, :],
            multimask_output=True,
        )
    
    max_score_index = np.argmax(scores)

    # Return the mask corresponding to the highest score
    return masks[max_score_index]
    
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
    ans_float = list(points[random.randint(0, len(points) - 1)])
    ans = [int(ans_float[0]), int(ans_float[1])]
    return np.array([ans])
    

def find_multiple_points(mask, num_points=5):
    points = [list(find_geometric_center(mask)[0])]
    while len(points) < num_points:
        point = find_random_point(mask)
        points.append(list(point[0]))
    print(points)
    return np.array(points)

def find_bounding_box(mask):
    objects = ndimage.find_objects(mask)
    if not objects:
        return None
    
    # Assuming you want the bounding box of the first object
    obj_slice = objects[0]

    # Extract bounding box coordinates
    y_min, y_max = obj_slice[0].start, obj_slice[0].stop
    x_min, x_max = obj_slice[1].start, obj_slice[1].stop

    # Return as (x_min, y_min, x_max, y_max)
    return np.array([x_min, y_min, x_max, y_max])
    
PATH = './ckpts/finetuned_sam.pth'
def train():
    lr = 1e-5
    wd = 0
    optimizer = torch.optim.Adam(sam_model.mask_decoder.parameters(), lr=lr, weight_decay=wd)
    epoch_num = 5
    logging.basicConfig(filename='training_log.log', level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    loss_fn = torch.nn.MSELoss()
    for epoch in range(epoch_num):
        epoch_loss = []
        epoch_start_time = time.time()
        for img_name, label_name in zip(val_img_names, val_label_names):
            print(img_name)
            image = nib.load('/home/jiaxin/ML23Fall/data/images/RawData/Training/imagesTr/' + img_name)
            label_image = nib.load('/home/jiaxin/ML23Fall/data/images/RawData/Training/labelsTr/' +label_name)
            image_data = image.get_fdata()
            label_data = label_image.get_fdata()
            # Get unique classes (excluding background)
            classes = np.unique(label_data)[1:]  # Excludes 0 if it represents the background
            for class_idx in classes:
                for slice_index in range(label_data.shape[2]):
                    with torch.no_grad(): 
                        slice_data = label_data[:, :, slice_index]
                        slice_image = get_image(image_data, slice_index)
                        #input_image = torch.from_numpy(slice_image).float()
                        
                        transform = ResizeLongestSide(sam_model.image_encoder.img_size)
                        input_image = transform.apply_image(slice_image)
                        input_image_torch = torch.as_tensor(input_image, device=device)
                        transformed_image = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
                        
                        input_image = sam_model.preprocess(transformed_image)
                        original_image_size = slice_image.shape[:2]
                        input_size = tuple(transformed_image.shape[-2:])

                    
                        mask = slice_data == class_idx
                        if not np.any(mask):
                            continue
                        image_embedding = sam_model.image_encoder(input_image)
                        input_point = torch.from_numpy(find_random_point(mask))
                        prompt_box = find_bounding_box(mask)
                        box = transform.apply_boxes(prompt_box, original_image_size)
                        box_torch = torch.as_tensor(box, dtype=torch.float, device=device)
                        box_torch = box_torch[None, :]
                        
                        sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                            points=None,
                            boxes=box_torch,
                            masks=None,
                        )
                        
                    low_res_masks, iou_predictions = sam_model.mask_decoder(
                        image_embeddings=image_embedding,
                        image_pe=sam_model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False,
                    )

                    upscaled_masks = sam_model.postprocess_masks(low_res_masks, input_size, original_image_size).to(device)
                    binary_mask = normalize(threshold(upscaled_masks, 0.0, 0))
                    print(upscaled_masks)   
                    gt_mask_resized = torch.from_numpy(np.resize(mask, (1, 1, mask.shape[0], mask.shape[1]))).to(device)
                    gt_binary_mask = torch.as_tensor(gt_mask_resized > 0, dtype=torch.float32)
                    
                    loss = loss_fn(binary_mask, gt_binary_mask)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_loss.append(loss.item())
        epoch_duration = time.time() - epoch_start_time
        avg_epoch_loss = sum(epoch_loss) / len(epoch_loss)
        
        # Logging at the end of each epoch
        logging.info(f"Epoch {epoch} completed. Average Loss: {avg_epoch_loss}, Time: {epoch_duration}s")
        torch.save(sam_model.state_dict(), PATH)    
    
train()