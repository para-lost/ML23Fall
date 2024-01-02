import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import argparse
import time
import glob

from monai.losses import DiceCELoss
from monai.data import load_decathlon_datalist, decollate_batch
from monai.transforms import AsDiscrete
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference

from model.Universal_model import Universal_model
from dataset.dataloader import get_loader
from utils import loss
from utils.utils import dice_score, TEMPLATE, ORGAN_NAME, merge_label, visualize_label, get_key, NUM_CLASS
from utils.utils import extract_topk_largest_candidates, organ_post_process, threshold_organ

from medpy.metric.binary import __surface_distances

# directly make args here
from argparse import Namespace

torch.multiprocessing.set_sharing_strategy('file_system')

def normalized_surface_dice(a: np.ndarray, b: np.ndarray, threshold: float, spacing: tuple = None, connectivity=1):
    """
    This implementation differs from the official surface dice implementation! These two are not comparable!!!!!
    The normalized surface dice is symmetric, so it should not matter whether a or b is the reference image
    This implementation natively supports 2D and 3D images. Whether other dimensions are supported depends on the
    __surface_distances implementation in medpy
    :param a: image 1, must have the same shape as b
    :param b: image 2, must have the same shape as a
    :param threshold: distances below this threshold will be counted as true positives. Threshold is in mm, not voxels!
    (if spacing = (1, 1(, 1)) then one voxel=1mm so the threshold is effectively in voxels)
    must be a tuple of len dimension(a)
    :param spacing: how many mm is one voxel in reality? Can be left at None, we then assume an isotropic spacing of 1mm
    :param connectivity: see scipy.ndimage.generate_binary_structure for more information. I suggest you leave that
    one alone
    :return:
    """
    assert all([i == j for i, j in zip(a.shape, b.shape)]), "a and b must have the same shape. a.shape= %s, " \
                                                            "b.shape= %s" % (str(a.shape), str(b.shape))
    if spacing is None:
        spacing = tuple([1 for _ in range(len(a.shape))])
    a_to_b = __surface_distances(a, b, spacing, connectivity)
    b_to_a = __surface_distances(b, a, spacing, connectivity)
 
    numel_a = len(a_to_b)
    numel_b = len(b_to_a)
 
    tp_a = np.sum(a_to_b <= threshold) / numel_a
    tp_b = np.sum(b_to_a <= threshold) / numel_b
 
    fp = np.sum(a_to_b > threshold) / numel_a
    fn = np.sum(b_to_a > threshold) / numel_b
 
    dc = (tp_a + tp_b) / (tp_a + tp_b + fp + fn + 1e-8)  # 1e-8 just so that we don't get div by 0
    return dc


# settings 
dist = False # bool, default=False,distributed training or not
local_rank = 1 # gpu
device = "cuda" 
epoch = 0
log_name = "Nvidia/new_clip/clip1_exo_extendedTrain" # The path resume from ckpt
start_epoch = 500 # Number of start epoches
end_epoch = 490 # Number of end epoches
epoch_interval = 100 # Number of start epoches
backbone = 'unet' # backbone [swinunetr or unet]
## hyperparameter
max_epoch = 1000 # Number of training epoches
store_num = 10 # Store model how often
lr = 1e-4 # Learning rate
weight_decay = 1e-5 # Weight Decay
## dataset
dataset_list = ['PAOT_123457891213','PAOT_10_inner'] 
data_root_path = '/home/jliu288/data/whole_organ/' # data root path
data_txt_path = './dataset/dataset_list/' # data txt path
batch_size = 1 # batch size
num_workers = 8 # workers numebr for DataLoader
a_min = -175 # a_min in ScaleIntensityRanged
a_max = 250 # a_max in ScaleIntensityRanged
b_min = 0.0 # b_min in ScaleIntensityRanged
b_max = 1.0 # b_max in ScaleIntensityRanged
space_x = 1.5 # spacing in x direction
space_y = 1.5 # spacing in y direction
space_z = 1.5 # spacing in z direction
roi_x = 96 #　roi size in x direction
roi_y = 96 # roi size in y direction
roi_z = 96 # roi size in z direction
num_samples =1 # sample number in each ct

phase = 'validation' # train or validation or test
cache_dataset = False # whether use cache dataset
cache_rate = 0.6 # The percentage of cached data in total
store_result = False #whether save prediction result
overlap = 0.5 # overlap for sliding_window_inference

args = Namespace(dist=dist, local_rank=local_rank, device=device, epoch=epoch, log_name=log_name,start_epoch=start_epoch,end_epoch=end_epoch,
epoch_interval=epoch_interval,backbone=backbone,max_epoch=max_epoch, store_num=store_num,lr=lr, weight_decay=weight_decay,dataset_list=dataset_list,
data_root_path=data_root_path,data_txt_path=data_txt_path,batch_size=batch_size,num_workers=num_workers,a_min=a_min,a_max=a_max,b_min=b_min,
b_max=b_max,space_x=space_x,space_y=space_y,space_z=space_z,roi_x=roi_x,roi_y=roi_y,roi_z=roi_z,num_samples=num_samples,phase=phase
cache_dataset=cache_dataset,cache_rate=cache_rate,store_result=store_result,overlap=overlap)


def validation(data_root_path ):
    data_root_path = data_root_path 
    # prepare the 3D model
    model = Universal_model(img_size=(roi_x, roi_y, roi_z),
                    in_channels=1,
                    out_channels=NUM_CLASS,
                    backbone=backbone,
                    encoding='word_embedding'
                    )
    #Load pre-trained weights
    store_path_root = 'out/Nvidia/new_clip/clip1_partialv2/epoch_***.pth'
    for store_path in glob.glob(store_path_root):
        # store_path = store_path_root
        store_dict = model.state_dict()
        load_dict = torch.load(store_path)['net']

        for key, value in load_dict.items():
            if 'swinViT' in key or 'encoder' in key or 'decoder' in key:
                name = '.'.join(key.split('.')[1:])
                name = 'backbone.' + name
            else:
                name = '.'.join(key.split('.')[1:])
            store_dict[name] = value

        model.load_state_dict(store_dict)
        print(f'Load {store_path} weights')
        model.cuda()
        torch.backends.cudnn.benchmark = True
        validation_loader, val_transforms = get_loader(args)
        i = int(store_path.split('_')[-1].split('.')[0])+1
        # MAIN　VARIATION
        model.eval()
        dice_list = {}
        nsd_list = {}
        for key in TEMPLATE.keys():
            dice_list[key] = np.zeros((2, NUM_CLASS)) # 1st row for dice, 2nd row for count
            nsd_list[key] = np.zeros((2, NUM_CLASS)) # 1st row for dice, 2nd row for count
        for index, batch in enumerate(tqdm(validation_loader)):
            # print('%d processd' % (index))
            image, label, name = batch["image"].cuda(), batch["post_label"], batch["name"]
            with torch.no_grad():
                pred = sliding_window_inference(image, (roi_x, roi_y, roi_z), 1, model, overlap=overlap, mode='gaussian')
                pred_sigmoid = F.sigmoid(pred)
            pred_hard = threshold_organ(pred_sigmoid)
            pred_hard = pred_hard.cpu()
            torch.cuda.empty_cache()
            B = pred_sigmoid.shape[0]
            for b in range(B):
                content = 'case%s| '%(name[b])
                template_key = get_key(name[b])
                organ_list = TEMPLATE[template_key]
                pred_hard_post = organ_post_process(pred_hard.numpy(), organ_list, log_name+'/'+name[0].split('/')[0]+'/'+name[0].split('/')[-1])
                pred_hard_post = torch.tensor(pred_hard_post)
                # pred_hard_post = pred_hard
                for organ in organ_list:
                    if torch.sum(label[b,organ-1,:,:,:]) != 0:
                        dice_organ, recall, precision = dice_score(pred_hard_post[b,organ-1,:,:,:].cuda(), label[b,organ-1,:,:,:].cuda())
                        dice_list[template_key][0][organ-1] += dice_organ.item()
                        dice_list[template_key][1][organ-1] += 1

                        content += '%s: %.4f, '%(ORGAN_NAME[organ-1], dice_organ.item())
                        print('%s: dice %.4f, recall %.4f, precision %.4f'%(ORGAN_NAME[organ-1], dice_organ.item(), recall.item(), precision.item()))
                print(content)

            torch.cuda.empty_cache()
        
        ave_organ_dice = np.zeros((2, NUM_CLASS))

        with open('out/'+log_name+f'/b_val_{i}.txt', 'w') as f:
            for key in TEMPLATE.keys():
                organ_list = TEMPLATE[key]
                content = 'Task%s| '%(key)
                # content1 = 'NSD Task%s| '%(key)
                for organ in organ_list:
                    dice = dice_list[key][0][organ-1] / dice_list[key][1][organ-1]
                    content += '%s: %.4f, '%(ORGAN_NAME[organ-1], dice)
                    ave_organ_dice[0][organ-1] += dice_list[key][0][organ-1]
                    ave_organ_dice[1][organ-1] += dice_list[key][1][organ-1]

                print(content)
                f.write(content)
                f.write('\n')
            content = 'Average | '
            for i in range(NUM_CLASS):
                content += '%s: %.4f, '%(ORGAN_NAME[i], ave_organ_dice[0][i] / ave_organ_dice[1][i])
            print(content)
            f.write(content)
            f.write('\n')

            print(np.mean(ave_organ_dice[0] / ave_organ_dice[1]))
            f.write('%s: %.4f, '%('average', np.mean(ave_organ_dice[0] / ave_organ_dice[1])))
            f.write('\n')





#python validation.py >> out/Nvidia/ablation_clip/clip2.txt