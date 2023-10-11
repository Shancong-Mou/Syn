# In this script, we implement the cut-paste method, the following 4 experiments can be conducted:
# 1. whole image augmentation only
# 2. cut-paste-only
# 3. cut-paste + clean background (cut-paste-balanced)
# 4. whole image augmentation images + cut-paste-only images + clean background images

import sys
from tqdm import tqdm
import os
import argparse
# defined command line options
parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, required=True)# root to store results, including trained models, images, results
parser.add_argument('--data_root', type=str, required=True)# root to data
parser.add_argument('--num_epochs', type=int, required=True)# number of epochs
parser.add_argument('--dice_weight', type=float, required=True)# loss metric: loss = dice_weight* dice + (1-dice_weight)*bce
parser.add_argument('--lr', type=float, required=True)# learning rate
parser.add_argument('--lr_schedule_step_size', type=int, required=True)# learning rate decay schedule step size 
parser.add_argument('--batch_size', type=int, required=True)# batch size
parser.add_argument('--rnd_seed', type=int, required=True)# random seed
parser.add_argument('--apply_augmentations',default = False, action='store_true')# apply augmentations to cut-paste method or not
parser.add_argument('--apply_poisson',default = False, action='store_true') 
# apply Poisson image composition to cut-paste method or not
parser.add_argument(
  "--aug_paras_float",  
  nargs="*",  
  type=float,
  default=[0.1, 1.9, 0.1, 1.9, 0.1, 1.9, 0.0, 0.5, 30.0, 0.3, 2.0],
) # default values of the augmentation prarmeters , used both for image-level aug and cut-paste

parser.add_argument('--lam_weights', 
  nargs="*",
  type=float,
  default=[1, 0.0, 0.0],  # default if nothing is provided
) # the weight for 3 loss terms, they are: 1. image_level_aug, 2. cut_paste, 3. clean background images 
parser.add_argument('--number_of_defect_per_image', type = int, default = 1) # number of defect pasted to target image in cut-paste method
parser.add_argument('--save_all_images', default = False, action='store_true') # save all iamges

# read all parameters
args = parser.parse_args()
num_epochs = args.num_epochs 
root = args.root
data_root = args.data_root
dice_weight = args.dice_weight 
lr = args.lr 
rnd_seed = args.rnd_seed
batch_size = args.batch_size 
lr_schedule_step_size = args.lr_schedule_step_size
br_l, br_u, cst_l, cst_u, sat_l, sat_u, hue_l, hue_u, d, s, scl_u = args.aug_paras_float
weight_image_level_aug, weight_defect_cut_paste, weight_good = args.lam_weights
number_of_defect_per_image = args.number_of_defect_per_image
apply_augmentations = args.apply_augmentations
apply_poisson = args.apply_poisson
save_all_images = args.save_all_images
model_save_folder = root

# import necessary libraries
import kornia as K
import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple
from diffAugs import _Jitter, _Rotation, _Shear, _Scale # augmentation parameters
from MVTecDataLoader_old import MVtechDataset_bkg, MVtechDataset_defect # MVtech data loader
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from AugmentPolicy import SubPolicyStage # augmentation policy
import os
import segmentation_models_pytorch as smp # smp pytorch segmentation models
import torch.optim as optim
import torch.nn.functional as F
from kornia.augmentation.auto import TrivialAugment # trivial augment 

# fix the random seed
import random
import numpy as np
torch.manual_seed(rnd_seed)
random.seed(rnd_seed)
np.random.seed(rnd_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(rnd_seed)
print(30*'-')
print('finish loading all modules: ')
print(30*'-')

# specify iamge storgation
# setup storage    
store_root = root 
os.makedirs(store_root, exist_ok= True)

image_root = store_root+'_rnd_seed_'+str(rnd_seed)+'_batch_size_'+str(batch_size)+'_num_epochs_'+str(num_epochs)+'_lr_'+str(lr)+ '_lr_schedule_step_size_'+str(lr_schedule_step_size) + '/'
os.makedirs(image_root, exist_ok= True)
model_root = store_root+'_rnd_seed_'+str(rnd_seed)+'_batch_size_'+str(batch_size)+'_num_epochs_'+str(num_epochs)+'_lr_'+str(lr)+ '_lr_schedule_step_size_'+str(lr_schedule_step_size) + 'model/'
os.makedirs(model_root, exist_ok= True)

## import Unet
from pprint import pprint
from torch.utils.data import DataLoader
# import a Unet backbone with imagenet pretrained weights
import segmentation_models_pytorch as smp

print(30*'-')
model = smp.Unet(
    encoder_name="resnet18",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=1,                      # model output channels (number of classes in your dataset)
)
print('imagenet pretrained model loaded')
print(30*'-')

# specify device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

from segmentation_models_pytorch.losses import DiceLoss
from torch import optim
# define batch size and numbe of defects per iamge
number_of_defect_per_image = 2 # two additional defects per target image
dim = (256,256) # image shape to be resized

# load data
# load clean background data
MVtechDataset_good_dataset = MVtechDataset_bkg(root_dir= "./data/" + data_root +"/good",dim =dim)
dataloader_good = DataLoader(MVtechDataset_good_dataset, batch_size=batch_size,
                        shuffle=True, num_workers=0, drop_last= True)
# load training data (with defects), those will aslo be used as target background for cut-paste
MVtechDataset_bkg_dataset = MVtechDataset_bkg(root_dir= "./data/" + data_root +"/combined/train", dim = dim,
                                                    mask_dir= "./data/" + data_root +"/combined/train_gt")
dataloader_bkg = DataLoader(MVtechDataset_bkg_dataset, batch_size=batch_size,
                        shuffle=True, num_workers=0, drop_last= True)

# load defect dataset (defects from the defect library, here is the same  dataset as the training set)
MVtechDataset_defect_dataset = MVtechDataset_defect(root_dir= "./data/" + data_root +"/combined/train", dim = dim,
                                                    mask_dir= "./data/" + data_root +"/combined/train_gt",
                                                    crop =True)
dataloader_defect = DataLoader(MVtechDataset_defect_dataset, batch_size=(batch_size)*number_of_defect_per_image,
                        shuffle=True, num_workers=0, drop_last= True)

# load val and test set
MVtechDataset_val_dataset = MVtechDataset_defect(root_dir= "./data/" + data_root +"/combined/val", 
                                                mask_dir= "./data/" + data_root +"/combined/val_gt",
                                                dim = dim)
dataloader_val = DataLoader(MVtechDataset_val_dataset, batch_size=batch_size,
                        shuffle=False, num_workers=12, drop_last= True)

MVtechDataset_test_dataset = MVtechDataset_defect(root_dir= "./data/" + data_root +"/combined/test", 
                                                mask_dir= "./data/" + data_root +"/combined/test_gt",
                                                dim = dim)
dataloader_test = DataLoader(MVtechDataset_test_dataset, batch_size=batch_size,
                        shuffle=False, num_workers=12, drop_last= True)

# iou calculation
def Prob_mask2SigmoidMask(prob_mask):# convert proba_mask tensor to between [0,1] 
    return (F.logsigmoid(prob_mask).exp() > 0.5).float()
def cal_iou(prob_mask, true_mask):
    # this calculates IoU scores
    pred_mask = Prob_mask2SigmoidMask(prob_mask)
    tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), true_mask.long(), mode="binary")
    per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise").item()
    return per_image_iou

# define loss function
bce = torch.nn.BCEWithLogitsLoss() # BCE loss
dice = DiceLoss("binary", classes=None, from_logits=True, smooth=0.0, ignore_index=None, eps=1e-07) # dice loss


############# defect augmentation section
# initilize the augmentation policy learnable range parameters with trival augment
_p = lambda x: torch.tensor(x, requires_grad= True,  device=device)
brightness = _p([br_l, br_u])
contrast = _p([cst_l, cst_u])
saturation = _p([sat_l, sat_u])
hue = _p([ hue_l, hue_u])

degrees = _p([-d, d]) # first optimzie geometric parameters [-180, 180]
shearX = _p([-s*dim[0], s*dim[0], 0.0, 0.0 ]) # [0,1]
shearY = _p([0.0, 0.0, -s*dim[1], s*dim[1] ]) # [0,1]
scale = _p([0.0, scl_u, 0.0, scl_u]) # [0, 10]

# intailize the policy
Jitter = _Jitter(brightness, contrast, saturation, hue)
Rotation = _Rotation(degrees)
ShearX = _Shear(shearX)
ShearY = _Shear(shearY)
Scale = _Scale(scale)
cut_paste_operations = [Jitter, Rotation, ShearX, ShearY, Scale]

policy = SubPolicyStage(operations = cut_paste_operations, apply_operations = apply_augmentations, apply_poisson = apply_poisson)

########
## image level augmentation section
# Trivial augmentation policy
default_policy = [
    # [("identity", 0, 1)],
    [("auto_contrast", 0, 1)],
    [("equalize", 0, 1)],
    [("rotate", -30.0, 30.0)],
    [("posterize", 0.0, 4)],
    [("solarize", 0.0, 1.0)],
    # (Color, 0.1, 1.9),
    [("contrast", 0.1, 1.9)],
    [("brightness", 0.1, 1.9)],
    [("sharpness", 0.1, 1.9)],
    [("shear_x", -0.3, 0.3)],
    [("shear_y", -0.3, 0.3)],
    # [("translate_x", -0.5, 0.5)],
    # [("translate_y", -0.5, 0.5)],
]

def parameterize_policy(policy):
    # convert all the values to torch tensors
    policy = [[(op, float(min_val), float(max_val)) for op, min_val, max_val in level] for level in policy]
    # convert min/max values to torch tensors
    policy = [[(op, _p(min_val), _p(max_val)) for op, min_val, max_val in level] for level in policy]
    # collect all the min/max values into a list
    values_w = [torch.stack([min_val, max_val]) for level in policy for op, min_val, max_val in level]
    return policy, values_w

policy_w, values_w = parameterize_policy(default_policy)
image_level_aug_policy = K.augmentation.AugmentationSequential(TrivialAugment(policy_w), data_keys=["input", "mask"])

# specify the optimizer and its parameters 
from torch.optim.lr_scheduler import StepLR
optimizer1 = optim.Adam(model.parameters(),lr=lr) # for segemntation model
optimizer1.zero_grad() 
scheduler = StepLR(optimizer1, step_size=lr_schedule_step_size, gamma=0.5)
# define epochs
epochs = num_epochs
train_loss = []
val_loss = []
test_loss = []
iter_num = 0

# esatblish the root for storing training images
train_img_root = store_root + 'train/'
os.makedirs(train_img_root, exist_ok= True)

for ep in range(epochs):
    # train loss
    train_loss_image_level = 0.0
    train_loss_bce = 0.0
    train_loss_dice = 0.0
    train_per_image_iou = 0.0

    # val loss
    val_loss_bce = 0.0
    val_loss_dice = 0.0  
    val_per_image_iou = 0.0

    # test loss
    test_loss_bce = 0.0
    test_loss_dice = 0.0  
    test_per_image_iou =0.0

    
    model.train()
    iterator = iter(dataloader_defect)
    iterator_good = iter(dataloader_good) # inject good images
    for i_batch, sample_batched_img_level in enumerate(tqdm(dataloader_bkg, desc = 'epoch' + str(ep) +'_train')): # use image_level_aug as main iteration
        sample_batched_defect = next(iterator, None)
        if sample_batched_defect == None:
            iterator = iter(dataloader_defect)
            sample_batched_defect = next(iterator, None)

        sample_batched_good = next(iterator_good, None)
        if sample_batched_good == None:
            iterator_good = iter(dataloader_good) # inject good images
            sample_batched_good = next(iterator_good, None)

        ###### calculate loss for cut-paste-augmented images tr_loss_defect
        defect_batches = torch.split(sample_batched_defect['defect_image'].clone(), batch_size)
        defect_batches_masks = torch.split(sample_batched_defect['defect_image_mask'].clone(), batch_size)
        for defects_round in range(number_of_defect_per_image):
            # print(defects_round)
            if defects_round == 0:
                input = [sample_batched_img_level['image'].clone(), sample_batched_img_level['img_mask'].clone(), defect_batches[defects_round], defect_batches_masks[defects_round], sample_batched_img_level['defect_location_mask'].clone() ]
            else:
                input = [output[0], output[1], defect_batches[defects_round], defect_batches_masks[defects_round], sample_batched_img_level['defect_location_mask'].clone() ]
            output = policy(input) # output is [ Tensor: augmented image, Tensor: Defect mask]
            augmented_image =  output[0].to(device)
            defect_mask = output[1][:,:1,:,:].to(device) # the output mask is C x H x W
        # forwrad
        pred_defect_mask = model(augmented_image.to(device))
        gt_defect_mask = defect_mask.to(device)[:,:1,:,:]
        tr_loss_defect = (1-dice_weight)*bce(pred_defect_mask, gt_defect_mask) +  dice_weight*dice(pred_defect_mask, gt_defect_mask)

    ###### calculate loss for clean background images tr_loss_img_good
        pred_mask_img_good = model(sample_batched_good['image'].to(device))
        gt_defect_mask_img_good = sample_batched_good['img_mask'].to(device)[:,:1,:,:]
        tr_loss_img_good = (1-dice_weight)*bce(pred_mask_img_good, gt_defect_mask_img_good) +  (dice_weight)*dice(pred_mask_img_good, gt_defect_mask_img_good)# input mask is 3 channels, convert to 


    ###### calculate loss for image level augmented images tr_loss_img_level
        augmented_img_level_image, augmented_img_level_image_mask = image_level_aug_policy(sample_batched_img_level['image'].clone(), sample_batched_img_level['img_mask']) 
        pred_mask_img_level = model(augmented_img_level_image.to(device))
        gt_defect_mask_img_level = augmented_img_level_image_mask.to(device)[:,:1,:,:]
        tr_loss_img_level = (1-dice_weight)*bce(pred_mask_img_level, gt_defect_mask_img_level) +  (dice_weight)*dice(pred_mask_img_level, gt_defect_mask_img_level)# input mask is 3 channels, convert to 

    ###### calculate total loss
        tr_loss = weight_image_level_aug*tr_loss_img_level + weight_defect_cut_paste*tr_loss_defect + weight_good*tr_loss_img_good

    # optimize         
        tr_loss.backward()
        optimizer1.step()
        optimizer1.zero_grad()

    # record train_loss
        train_loss_image_level = train_loss_image_level + tr_loss_img_level.detach().item()
        # record train_loss on orginal images (after img_level aug) metrics, including: bce, dice and IoU
        train_loss_bce = train_loss_bce + bce(pred_mask_img_level, gt_defect_mask_img_level).detach().item()
        train_loss_dice = train_loss_dice +  dice(pred_mask_img_level, gt_defect_mask_img_level).detach().item()
        train_per_image_iou = train_per_image_iou + cal_iou(pred_mask_img_level, gt_defect_mask_img_level)

    # print(train_loss)
    if ep%10 == 0: # store trained model after every 10 epochs
        torch.save(model, model_root+'model'+str(ep))

    train_loss_image_level = train_loss_image_level/len(dataloader_bkg)
    train_loss_dice = train_loss_dice/len(dataloader_bkg)
    train_loss_bce = train_loss_bce/len(dataloader_bkg)
    train_per_image_iou = train_per_image_iou/len(dataloader_bkg)
    train_loss.append([train_loss_bce, train_loss_dice, train_per_image_iou, train_loss_image_level])
    scheduler.step()

    # # evaluate
    # model.eval()
    # with torch.no_grad():
    # val
    for k_batch, sample_val in enumerate(tqdm(dataloader_val, desc = 'epoch' + str(ep) +'_val')):
        pred_mask = model(sample_val['defect_image'].to(device))
        gt_defect_mask = sample_val['defect_image_mask'][:,:1,:,:].to(device)
        val_loss_bce = val_loss_bce + bce(pred_mask, gt_defect_mask).item()
        val_loss_dice = val_loss_dice+  dice(pred_mask, gt_defect_mask).item()
        val_per_image_iou = val_per_image_iou + cal_iou(pred_mask, gt_defect_mask)
    val_loss_dice = val_loss_dice/len(dataloader_val)
    val_loss_bce = val_loss_bce/len(dataloader_val)
    val_per_image_iou = val_per_image_iou/len(dataloader_val)
    val_loss.append([val_loss_bce, val_loss_dice, val_per_image_iou])
    
    # test
    for k_batch, sample_test in enumerate(tqdm(dataloader_test, desc = 'epoch' + str(ep) +'_test')):
        pred_mask = model(sample_test['defect_image'].to(device))
        gt_defect_mask = sample_test['defect_image_mask'][:,:1,:,:].to(device)
        test_loss_bce =test_loss_bce +  bce(pred_mask, gt_defect_mask).item()
        test_loss_dice =test_loss_dice + dice(pred_mask, gt_defect_mask).item()
        test_per_image_iou = test_per_image_iou + cal_iou(pred_mask, gt_defect_mask)
    test_loss_dice = test_loss_dice/len(dataloader_test)
    test_loss_bce = test_loss_bce/len(dataloader_test)
    test_per_image_iou = test_per_image_iou/len(dataloader_test)
    test_loss.append([test_loss_bce, test_loss_dice, test_per_image_iou])
          
    print("ep", ep, "train_loss", f"{train_loss_bce:.3}", f"{train_loss_dice:.3}", f"{train_per_image_iou:.3}", "val_loss", f"{val_loss_bce:.3}", f"{val_loss_dice:.3}",f"{val_per_image_iou:.3}", "test_loss", f"{test_loss_bce:.3}", f"{test_loss_dice:.3}", f"{test_per_image_iou:.3}", )


res = {}
res['train_loss'] = train_loss
res['val_loss'] = val_loss
res['test_loss'] = test_loss


# save results
import json
os.makedirs(store_root+'quant_resut/', exist_ok= True)
with open(store_root+'quant_resut/'+'_rnd_seed_'+str(rnd_seed)+'_btsz_'+str(batch_size)+'_epcs_'+str(num_epochs)+'_lr_'+str(lr)+'_l1_'+ str(weight_image_level_aug)+'_l2_'+ str(weight_defect_cut_paste)+'_l3_'+ str(weight_good)+'.json', 'w') as f:
    json.dump(res, f)

