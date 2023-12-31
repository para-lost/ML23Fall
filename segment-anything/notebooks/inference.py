import nibabel as nib
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
import os
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from PIL import Image
image = nib.load('/home/jiaxin/ML23Fall/data/images/RawData/Training/img/img0001.nii.gz')
label_image = nib.load('/home/jiaxin/ML23Fall/data/images/RawData/Training/label/label0001.nii.gz')
image_data = image.get_fdata()
label_data = label_image.get_fdata()
print(image_data)

def show_anns(anns, output_dir='./output/', filename='image.png'):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the figure
    plt.savefig(os.path.join(output_dir, filename))

def get_anns_score(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['predicted_iou']), reverse=True)
    sorted_anns = sorted_anns[:13]
    print(len(sorted_anns))
    ann_avg =  sorted_anns[0]['segmentation'].astype(np.float32)
    # for ann in sorted_anns[1:]:
    #     ann_avg += ann['segmentation'].astype(np.float32)
    #ann_avg /= len(sorted_anns)
    return ann_avg


    
def get_image_list(data):
    image_list = []
    # Loop through each slice
    for i in range(data.shape[-1]):
        # if i == 5:
        #     break
        # Extract the i-th slice (2D array)
        slice_data = data[:, :, i:i+1]
        # Normalize the slice to 0-255 range
        slice_normalized = cv2.normalize(slice_data, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

        # Convert to uint8
        slice_normalized = slice_normalized.astype(np.uint8)

        # Convert grayscale to pseudo-RGB by replicating the channel
        pseudo_rgb = cv2.cvtColor(slice_normalized, cv2.COLOR_GRAY2RGB)
        # pseudo_rgb = slice_normalized
        image_list.append(pseudo_rgb)
        print(pseudo_rgb)
        print(np.sum(pseudo_rgb))
    return image_list

    
def mdice(true_segmentation, predicted_segmentation):
    # print(true_segmentation)
    predicted_segmentation = torch.tensor(predicted_segmentation)
    intersection = torch.sum(true_segmentation == predicted_segmentation)
    print("true", torch.sum(true_segmentation))
    print("predicted", torch.sum(predicted_segmentation))
    return 2. * intersection / (torch.sum(true_segmentation == true_segmentation) + torch.sum(predicted_segmentation == predicted_segmentation))

def get_model():
    sam_checkpoint = "../sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(sam)
    return mask_generator
    
mask_generator = get_model()
image_list = get_image_list(image_data)
mask_list = []
for img in image_list:
    mask = mask_generator.generate(img)
    final_mask = get_anns_score(mask)
    print(np.sum(final_mask))
    mask_list.append((torch.from_numpy(final_mask)>0.).float())
print(mask_list)
masks = torch.stack(mask_list, dim=2)
print(torch.sum(masks))
# label_list_numpy = get_image_list(label_data)
# print("sum is:", torch.sum(torch.from_numpy(label_data)))
# print(label_list_numpy)
# label_list = []
# for label in label_list_numpy:
#     label_list.append(torch.from_numpy(label).float())
# labels = torch.stack(label_list, dim=2)
print(mdice(masks, label_data))