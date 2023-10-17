# In this script, we try to optimzie the weight of each augmentations 
# \sum_j \lambda_j \sum_{i_j} L(x_{i_j},y_{i_j})
# Where:
# image level aug images
# cut-paste-aug photometric
# cut-paste-aug Rotation
# cut-paste-aug Shear
# cut-paste-aug Scale
# clean background images

import sys
from tqdm import tqdm
import os
import argparse
import kornia as K
import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple
from diffAugs import _Jitter, _Rotation, _Shear, _Scale
from MVTecDataLoader_old import MVtechDataset_bkg, MVtechDataset_defect
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from AugmentPolicy import SubPolicyStage
import os
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from util import gather_flat_grad
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import grad
from torch.autograd import Variable
from kornia.augmentation.auto import TrivialAugment

# fix the random seed
import random
import numpy as np
# defined command line options
parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, required=True)# root to store
parser.add_argument('--data_root', type=str, required=True)# root to data
parser.add_argument('--num_epochs', type=int, required=True)# number of epochs
parser.add_argument('--dice_weight', type=float, required=True)# loss: dice or bce
parser.add_argument('--lr', type=float, required=True)# learning rate
parser.add_argument('--lr_schedule_step_size', type=int, required=True)# learning rate schedule 
parser.add_argument('--batch_size', type=int, required=True)# batch size
parser.add_argument('--rnd_seed', type=int, required=True)# random seed
parser.add_argument('--record_each_catgy', default = False, action='store_true')# record the loss function for each category
parser.add_argument('--train_from_scrach', default = False, action='store_true')# default use pretrained model
parser.add_argument('--warm_up', type=int, required=True) # if we do not use pretrained model, what is the warm_up steps

parser.add_argument('--apply_augmentations',default = False, action='store_true')# apply _aug
parser.add_argument('--apply_poisson',default = False, action='store_true')# 

parser.add_argument(
  "--aug_paras_float",
  nargs="*",  
  type=float,
  default=[0.1, 1.9, 0.1, 1.9, 0.1, 1.9, 0.0, 0.5, 30.0, 0.3, 2.0],
) # augmentation parameters

parser.add_argument('--lam_weights', 
  nargs="*",  # 0 or more values expected => creates a list
  type=float,
  default=[1, 0.0, 0.0, 0.0, 0.0, 0.0],  # default if nothing is provided
) 
# the weight for 6 loss terms, they are: 1. image_level_aug, 2. defect1_cut_paste, 3. defect2_cut_paste, 4 defect3_cut_paste, 5 defect4_cut_paste

parser.add_argument('--optimize_cut_paste_policy_parameter', default = False, action='store_true')# record the loss function for each category
parser.add_argument('--optimize_image_level_aug_policy_parameter', default = False, action='store_true')
parser.add_argument('--optimize_weights', default = False, action='store_true')

parser.add_argument('--number_of_defect_per_image', type = int, default = 1) # number of defect per image

parser.add_argument('--paste_to_all', default = False, action='store_true') # paste to all iamges 
parser.add_argument('--paste_to_clean', default = False, action='store_true') # paste to clean image only
parser.add_argument('--paste_per_cat', default = False, action='store_true') # paste to its own categories only
parser.add_argument('--save_all_images', default = False, action='store_true') # save all iamges



# parse the command line
args = parser.parse_args()
print(30*'-')
print('finish reading all parameters: ', args)
print(30*'-')

num_epochs = args.num_epochs # number of epochs
root = args.root # root to store
data_root = args.data_root # root to data
dice_weight = args.dice_weight # loss metric: dice or bce
lr = args.lr # learning rate
rnd_seed = args.rnd_seed# random seed
batch_size = args.batch_size #batch size
lr_schedule_step_size = args.lr_schedule_step_size # step size
record_each_catgy = args.record_each_catgy
br_l_, br_u_, cst_l_, cst_u_, sat_l_, sat_u_, hue_l_, hue_u_, d_, s_, scl_u_ = args.aug_paras_float
train_from_scrach = args.train_from_scrach
warm_up = args.warm_up
weight_image_level_aug, weight_defect_cut_paste_photometirc, weight_defect_cut_paste_rotate, weight_defect_cut_paste_shear, weight_defect_cut_paste_scale, weight_good = args.lam_weights
optimize_cut_paste_policy_parameter = args.optimize_cut_paste_policy_parameter
optimize_image_level_aug_policy_parameter = args.optimize_image_level_aug_policy_parameter
optimize_weights = args.optimize_weights
number_of_defect_per_image = args.number_of_defect_per_image
paste_to_all = args.paste_to_all
paste_to_clean = args.paste_to_clean
paste_per_cat = args.paste_per_cat
apply_augmentations = args.apply_augmentations
apply_poisson = args.apply_poisson
save_all_images = args.save_all_images
model_save_folder = root


torch.manual_seed(rnd_seed)
random.seed(rnd_seed)
np.random.seed(rnd_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(rnd_seed)
print(30*'-')
print('finish loading all modules: ')
print(30*'-')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def neumann_hyperstep_preconditioner(d_val_loss_d_theta, d_train_loss_d_w, elementary_lr, num_neumann_terms, trainable_params):
    preconditioner = d_val_loss_d_theta.detach()
    counter = preconditioner

    # Do the fixed point iteration to approximate the vector-inverseHessian product
    i = 0
    while i < num_neumann_terms:  # for i in range(num_neumann_terms):
        old_counter = counter

        # This increments counter to counter * (I - hessian) = counter - counter * hessian
        hessian_term = gather_flat_grad(
            grad(d_train_loss_d_w, trainable_params, grad_outputs=counter.view(-1), retain_graph=True))
        counter = old_counter - elementary_lr * hessian_term

        preconditioner = preconditioner + counter
        i += 1
    # return elementary_lr * preconditioner
    return preconditioner


def zero_hypergrad(get_hyper_train):
    """

    :param get_hyper_train:
    :return:
    """
    for p in get_hyper_train():
        p_num_params = np.prod(p.shape)
        if p.grad is not None:
            p.grad = p.grad * 0

def clip_hyper_train(get_hyper_train):
    """

    :param get_hyper_train:
    :return: clip the variabel in between [0.0, 10.0]
    """
    if optimize_weights:
        for p in get_hyper_train():
            p.data.clamp_(min=0.0, max=10.0)# notice that clamp p.data is necessary
        return 
    if optimize_cut_paste_policy_parameter:
        num = 0
        for p in get_hyper_train():
            if num <= 2: # first 3
                p.data.clamp_(0.0, 3.0)
            elif num <= 3: # hue
                p.data.clamp_(-0.5, 0.5)
            elif num <= 4: # degree
                p.data.clamp_(-180.0, 180.0)
            elif num <= 5: # shear
                p.data.clamp_(torch.tensor([-0.5*dim[0], -0.5*dim[0], 0.0, 0.0], device = device), torch.tensor([0.5*dim[0], 0.5*dim[0], 0.0, 0.0], device = device))
            elif num <= 6: # shear
                p.data.clamp_(torch.tensor([ 0.0, 0.0, -0.5*dim[1], -0.5*dim[1]], device = device), torch.tensor([0.0, 0.0, 0.5*dim[1], 0.5*dim[1]], device = device) )
            else:
                p.data.clamp_(torch.tensor([0.0, 0.0, 0.0, 0.0], device = device), torch.tensor([3.0, 3.0, 3.0, 3.0], device = device) )
            num= num + 1
        return
        
def store_hypergrad(get_hyper_train, get_hyper_train_grad):
    """

    :param get_hyper_train:
    :param total_d_val_loss_d_lambda:
    :return:
    """
    current_index = 0
    for index, p in enumerate(get_hyper_train()):
        p.grad = get_hyper_train_grad[index]

# specify iamge storgation
# setup storage    
store_root = root #'./results/Whole_image_aug/'
os.makedirs(store_root, exist_ok= True)

image_root = store_root+'_rnd_seed_'+str(rnd_seed)+'_batch_size_'+str(batch_size)+'_num_epochs_'+str(num_epochs)+'_lr_'+str(lr)+ '_lr_schedule_step_size_'+str(lr_schedule_step_size) + '/'
os.makedirs(image_root, exist_ok= True)
model_root = store_root+'_rnd_seed_'+str(rnd_seed)+'_batch_size_'+str(batch_size)+'_num_epochs_'+str(num_epochs)+'_lr_'+str(lr)+ '_lr_schedule_step_size_'+str(lr_schedule_step_size) + 'model/'
os.makedirs(model_root, exist_ok= True)

## import Unet
from torch.utils.data import DataLoader
# import a Unet backbone with imagenet pretrained weights
import segmentation_models_pytorch as smp
model_saving_path = './Defect_aug_with_default_policy_add_clean_only_parameter_MVTech_baseline/_rnd_seed_'+str(rnd_seed)+'_batch_size_2_num_epochs_150_lr_0.0005_lr_schedule_step_size_20model/model30'

print(30*'-')
if train_from_scrach:
    model = smp.Unet(
        encoder_name="resnet18",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=1,                      # model output channels (number of classes in your dataset)
    )
    print('imagenet pretrained model loaded')
else:
    model = torch.load(model_saving_path) # load pretrained model
    print('whole image augmentation pretrained model loaded')
print(30*'-')


model.to(device)

from segmentation_models_pytorch.losses import DiceLoss
from torch import optim
# define batch size and numbe of defects per iamge
number_of_defect_per_image = 2 
dim = (256,256) # image shape to be resized

# load data
# load background dataset # use additional clean background
MVtechDataset_good_dataset = MVtechDataset_bkg(root_dir= "./data/" + data_root +"/good",dim =dim)
dataloader_good = DataLoader(MVtechDataset_good_dataset, batch_size=batch_size,
                        shuffle=True, num_workers=0, drop_last= True)
# # load background dataset # use the training data set as background
MVtechDataset_bkg_dataset = MVtechDataset_bkg(root_dir= "./data/" + data_root +"/combined/train", dim = dim,
                                                    mask_dir= "./data/" + data_root +"/combined/train_gt")
dataloader_bkg = DataLoader(MVtechDataset_bkg_dataset, batch_size=batch_size,
                        shuffle=True, num_workers=0, drop_last= True)

# load defect dataset (defects from the training set)
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
    # prob_mask = model(sample_val['defect_image'].to(device)).sigmoid()
    pred_mask = Prob_mask2SigmoidMask(prob_mask)
    tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), true_mask.long(), mode="binary")
    per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise").item()
    # dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro").item()
    # f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro").item()
    # return per_image_iou, dataset_iou
    return per_image_iou#, dataset_iou, f1_score

# define loss function
bce = torch.nn.BCEWithLogitsLoss() # BCE loss
dice = DiceLoss("binary", classes=None, from_logits=True, smooth=0.0, ignore_index=None, eps=1e-07)


############# defect augmentation section
# initilize the augmentation policy learnable range parameters with trival augment
# specify device
# _p = lambda x: torch.tensor(x, requires_grad= True,  device=device)
t = lambda x: torch.tensor(x,device = device)
_p = lambda x: nn.Parameter(t(x))

brightness = _p([0.1, 1.9])
contrast = _p([0.1, 1.9])
saturation = _p([0.1, 1.9])
hue = _p([0.0, 0.0])
degrees = _p([-30.0, 30.0])
shearX = _p([-0.3*dim[0], 0.3*dim[0], 0.0, 0.0 ])
shearY = _p([0.0, 0.0, -0.3*dim[1], 0.3*dim[1] ])
scale = _p([0.0, 2.0, 0.0, 2.0])


# intailize the policy
Jitter = _Jitter(brightness, contrast, saturation, hue)
Rotation = _Rotation(degrees)
ShearX = _Shear(shearX)
ShearY = _Shear(shearY)
Scale = _Scale(scale)
cut_paste_operations_photometric = [Jitter]

cut_paste_operations_rotate = [Rotation]

cut_paste_operations_shear = [ShearX, ShearY]

cut_paste_operations_scale = [Scale]

policy_photometric = SubPolicyStage(operations = cut_paste_operations_photometric, apply_operations = apply_augmentations, apply_poisson = apply_poisson)

policy_rotate = SubPolicyStage(operations = cut_paste_operations_rotate, apply_operations = apply_augmentations, apply_poisson = apply_poisson)

policy_shear = SubPolicyStage(operations = cut_paste_operations_shear, apply_operations = apply_augmentations, apply_poisson = apply_poisson)

policy_scale = SubPolicyStage(operations = cut_paste_operations_scale, apply_operations = apply_augmentations, apply_poisson = apply_poisson)


########
## image level augmentation  section
# learnable trivial policy, starting from the default policy of trivial augment
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
    learnable_values = [torch.stack([min_val, max_val]) for level in policy for op, min_val, max_val in level]
    return policy, learnable_values

learnable_policy, learnable_values = parameterize_policy(default_policy)
image_level_aug_policy = K.augmentation.AugmentationSequential(TrivialAugment(learnable_policy), data_keys=["input", "mask"])

###### define optimizer
hyper_parameters = []

if optimize_cut_paste_policy_parameter:
    for operation in cut_paste_operations:
        hyper_parameters = hyper_parameters + list(operation.parameters())
    # hyper_parameters = list([br_l, br_u, cst_l, cst_u, sat_l, sat_u, hue_l, hue_u, d, s, scl_u])

# for tens in hyper_parameters:       
#     print(tens.is_leaf)


if optimize_image_level_aug_policy_parameter:
    hyper_parameters = hyper_parameters+ list(learnable_values)

if optimize_weights:
    lam_defect_cut_paste_scale = _p(weight_defect_cut_paste_scale)
    lam_defect_cut_paste_rotate = _p(weight_defect_cut_paste_rotate)
    lam_defect_cut_paste_shear = _p(weight_defect_cut_paste_shear)
    lam_defect_cut_paste_photometirc = _p(weight_defect_cut_paste_photometirc)
    lam_good = _p(weight_good)
    hyper_parameters = hyper_parameters + [lam_defect_cut_paste_photometirc, lam_defect_cut_paste_rotate, lam_defect_cut_paste_scale, lam_defect_cut_paste_shear, lam_good]


    
def get_hyper_train():
    return hyper_parameters

def get_hyper_train_flat():
    return torch.cat([p.view(-1) for p in hyper_parameters])

from torch.optim.lr_scheduler import StepLR
hyper_optimizer = optim.Adam(get_hyper_train(), lr = 1e-1)
hyper_scheduler =StepLR(hyper_optimizer, step_size=lr_schedule_step_size, gamma=0.5)

optimizer1 = optim.Adam(model.parameters(),lr=lr) # for segemntation model
optimizer1.zero_grad() 
scheduler = StepLR(optimizer1, step_size=lr_schedule_step_size, gamma=0.5)
# define epochs
epochs = num_epochs
train_loss = []
val_loss = []
test_loss = []
# each category loss
val_loss_per_catg = []
test_loss_per_catg = []

iter_num = 0

train_img_root = store_root + 'train/'
os.makedirs(train_img_root, exist_ok= True)

for ep in range(epochs):
    # train loss
    train_loss_image_level = 0.0
    train_loss_defect1 = 0.0
    train_loss_defect2 = 0.0
    train_loss_defect3 = 0.0
    train_loss_defect4 = 0.0
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

    
    zero_hypergrad(get_hyper_train)
    NUM_ACCUMULATION_STEPS = 1
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

        ###### calculate loss for augmented images tr_loss_defect_photometric
        # augment: apply the defect mutiple times
            # augment apply the defect mutiple times
        defect_batches = torch.split(sample_batched_defect['defect_image'].clone(), batch_size)
        defect_batches_masks = torch.split(sample_batched_defect['defect_image_mask'].clone(), batch_size)
        for defects_round in range(number_of_defect_per_image):
            # print(defects_round)
            if defects_round == 0:
                input = [sample_batched_img_level['image'].clone(), sample_batched_img_level['img_mask'].clone(), defect_batches[defects_round], defect_batches_masks[defects_round], sample_batched_img_level['defect_location_mask'].clone() ]
            else:
                input = [output_photometric[0], output_photometric[1], defect_batches[defects_round], defect_batches_masks[defects_round], sample_batched_img_level['defect_location_mask'].clone() ]
            # with torch.no_grad(): if not learning the policy, then we can keep this with no grad
            output_photometric = policy_photometric(input) # output_photometric is [ Tensor: augmented image, Tensor: Defect mask]
            augmented_image_photometric =  output_photometric[0].to(device)
            defect_mask_photometric = output_photometric[1][:,:1,:,:].to(device) # the output_photometric mask is C x H x W
        # forwrad
        pred_defect_mask_photometric = model(augmented_image_photometric.to(device))
        gt_defect_mask_photometric = defect_mask_photometric.to(device)[:,:1,:,:]
        tr_loss_defect_photometric = (1-dice_weight)*bce(pred_defect_mask_photometric, gt_defect_mask_photometric) +  dice_weight*dice(pred_defect_mask_photometric, gt_defect_mask_photometric)

        ###### calculate loss for augmented images tr_loss_defect_shear
        # augment: apply the defect mutiple times
            # augment apply the defect mutiple times
        defect_batches = torch.split(sample_batched_defect['defect_image'].clone(), batch_size)
        defect_batches_masks = torch.split(sample_batched_defect['defect_image_mask'].clone(), batch_size)
        for defects_round in range(number_of_defect_per_image):
            # print(defects_round)
            if defects_round == 0:
                input = [sample_batched_img_level['image'].clone(), sample_batched_img_level['img_mask'].clone(), defect_batches[defects_round], defect_batches_masks[defects_round], sample_batched_img_level['defect_location_mask'].clone() ]
            else:
                input = [output_shear[0], output_shear[1], defect_batches[defects_round], defect_batches_masks[defects_round], sample_batched_img_level['defect_location_mask'].clone() ]
            # with torch.no_grad(): if not learning the policy, then we can keep this with no grad
            output_shear = policy_shear(input) # output_shear is [ Tensor: augmented image, Tensor: Defect mask]
            augmented_image_shear =  output_shear[0].to(device)
            defect_mask_shear = output_shear[1][:,:1,:,:].to(device) # the output_shear mask is C x H x W
        # forwrad
        pred_defect_mask_shear = model(augmented_image_shear.to(device))
        gt_defect_mask_shear = defect_mask_shear.to(device)[:,:1,:,:]
        tr_loss_defect_shear = (1-dice_weight)*bce(pred_defect_mask_shear, gt_defect_mask_shear) +  dice_weight*dice(pred_defect_mask_shear, gt_defect_mask_shear)

        ###### calculate loss for augmented images tr_loss_defect_rotate
        # augment: apply the defect mutiple times
            # augment apply the defect mutiple times
        defect_batches = torch.split(sample_batched_defect['defect_image'].clone(), batch_size)
        defect_batches_masks = torch.split(sample_batched_defect['defect_image_mask'].clone(), batch_size)
        for defects_round in range(number_of_defect_per_image):
            # print(defects_round)
            if defects_round == 0:
                input = [sample_batched_img_level['image'].clone(), sample_batched_img_level['img_mask'].clone(), defect_batches[defects_round], defect_batches_masks[defects_round], sample_batched_img_level['defect_location_mask'].clone() ]
            else:
                input = [output_rotate[0], output_rotate[1], defect_batches[defects_round], defect_batches_masks[defects_round], sample_batched_img_level['defect_location_mask'].clone() ]
            # with torch.no_grad(): if not learning the policy, then we can keep this with no grad
            output_rotate = policy_rotate(input) # output_rotate is [ Tensor: augmented image, Tensor: Defect mask]
            augmented_image_rotate =  output_rotate[0].to(device)
            defect_mask_rotate = output_rotate[1][:,:1,:,:].to(device) # the output_rotate mask is C x H x W
        # forwrad
        pred_defect_mask_rotate = model(augmented_image_rotate.to(device))
        gt_defect_mask_rotate = defect_mask_rotate.to(device)[:,:1,:,:]
        tr_loss_defect_rotate = (1-dice_weight)*bce(pred_defect_mask_rotate, gt_defect_mask_rotate) +  dice_weight*dice(pred_defect_mask_rotate, gt_defect_mask_rotate)

        ###### calculate loss for augmented images tr_loss_defect_scale
        # augment: apply the defect mutiple times
            # augment apply the defect mutiple times
        defect_batches = torch.split(sample_batched_defect['defect_image'].clone(), batch_size)
        defect_batches_masks = torch.split(sample_batched_defect['defect_image_mask'].clone(), batch_size)
        for defects_round in range(number_of_defect_per_image):
            # print(defects_round)
            if defects_round == 0:
                input = [sample_batched_img_level['image'].clone(), sample_batched_img_level['img_mask'].clone(), defect_batches[defects_round], defect_batches_masks[defects_round], sample_batched_img_level['defect_location_mask'].clone() ]
            else:
                input = [output_scale[0], output_scale[1], defect_batches[defects_round], defect_batches_masks[defects_round], sample_batched_img_level['defect_location_mask'].clone() ]
            # with torch.no_grad(): if not learning the policy, then we can keep this with no grad
            output_scale = policy_scale(input) # output_scale is [ Tensor: augmented image, Tensor: Defect mask]
            augmented_image_scale =  output_scale[0].to(device)
            defect_mask_scale = output_scale[1][:,:1,:,:].to(device) # the output_scale mask is C x H x W
        # forwrad
        pred_defect_mask_scale = model(augmented_image_scale.to(device))
        gt_defect_mask_scale = defect_mask_scale.to(device)[:,:1,:,:]
        tr_loss_defect_scale = (1-dice_weight)*bce(pred_defect_mask_scale, gt_defect_mask_scale) +  dice_weight*dice(pred_defect_mask_scale, gt_defect_mask_scale)

    ###### calculate loss for good augmented images tr_loss_img_level
        pred_mask_img_good = model(sample_batched_good['image'].to(device))
        gt_defect_mask_img_good = sample_batched_good['img_mask'].to(device)[:,:1,:,:]
        tr_loss_img_good = (1-dice_weight)*bce(pred_mask_img_good, gt_defect_mask_img_good) +  (dice_weight)*dice(pred_mask_img_good, gt_defect_mask_img_good)# input mask is 3 channels, convert to 


    ###### calculate loss for image level augmented images tr_loss_img_level
        augmented_img_level_image, augmented_img_level_image_mask = image_level_aug_policy(sample_batched_img_level['image'].clone(), sample_batched_img_level['img_mask']) 
        pred_mask_img_level = model(augmented_img_level_image.to(device))
        gt_defect_mask_img_level = augmented_img_level_image_mask.to(device)[:,:1,:,:]
        tr_loss_img_level = (1-dice_weight)*bce(pred_mask_img_level, gt_defect_mask_img_level) +  (dice_weight)*dice(pred_mask_img_level, gt_defect_mask_img_level)# input mask is 3 channels, convert to 

    ###### calculate total loss
        if optimize_weights:
            tr_loss = weight_image_level_aug*tr_loss_img_level + lam_defect_cut_paste_photometirc*tr_loss_defect_photometric + lam_defect_cut_paste_rotate*tr_loss_defect_rotate + lam_defect_cut_paste_shear*tr_loss_defect_shear + lam_defect_cut_paste_scale*tr_loss_defect_scale + lam_good*tr_loss_img_good
        else:
            tr_loss = weight_image_level_aug*tr_loss_img_level + weight_defect_cut_paste_photometirc*tr_loss_defect_photometric + weight_defect_cut_paste_rotate*tr_loss_defect_rotate + weight_defect_cut_paste_shear*tr_loss_defect_shear + weight_defect_cut_paste_scale*tr_loss_defect_scale + weight_good*tr_loss_img_good

    # optimize
        tr_loss /= NUM_ACCUMULATION_STEPS
         
        tr_loss.backward()
        if i_batch == 0:
            optimizer1.step()
            trainable_params = [p for p in model.parameters() if p.grad is not None]
            num_weights, num_hypers = sum(p.numel() for p in model.parameters() if p.grad is not None), sum(p.numel() for p in get_hyper_train())           
        if ((i_batch + 1) % NUM_ACCUMULATION_STEPS == 0) or (i_batch + 1 == len(dataloader_bkg)):
            optimizer1.step()
            optimizer1.zero_grad()

    # record train_loss
        train_loss_image_level = train_loss_image_level + tr_loss_img_level.detach().item()
        # record train_loss on orginal images (after img_level aug) metrics, including: bce, dice and IoU
        train_loss_bce = train_loss_bce + bce(pred_mask_img_level, gt_defect_mask_img_level).detach().item()
        train_loss_dice = train_loss_dice +  dice(pred_mask_img_level, gt_defect_mask_img_level).detach().item()
        train_per_image_iou = train_per_image_iou + cal_iou(pred_mask_img_level, gt_defect_mask_img_level)
    # hyper_step
        # print(f"num_weights : {num_weights}, num_hypers : {num_hypers}")
        if ep>=warm_up:# and ep%5 == 0:
            if i_batch%7 == 0:
                # accumulate hypergrad
                num_hyper_grad_accumulate = 1
                hyper_grad = 0
                for hyper_step in range(num_hyper_grad_accumulate):
                    model.train(), model.zero_grad()
                    # calculate d_train_d_w
                    num_tr_accumulate = 5
                    d_train_loss_d_w = torch.zeros(num_weights).to(device)
                    
                    iterator = iter(dataloader_defect)
                    iterator_good = iter(dataloader_good) # inject good images
                    for tr_batch, sample_batched_img_level in enumerate(dataloader_bkg): # use image_level_aug as main iteration
                    # load defect1: 'sample_batched_defect1' and corresponding background: 'sample_defect1_bkg'
                        sample_batched_defect = next(iterator, None)
                        if sample_batched_defect == None:
                            iterator = iter(dataloader_defect)
                            sample_batched_defect = next(iterator, None)

                        sample_batched_good = next(iterator_good, None)
                        if sample_batched_good == None:
                            iterator_good = iter(dataloader_good) # inject good images
                            sample_batched_good = next(iterator_good, None)


                        ###### calculate loss for augmented images tr_loss_defect_photometric
                        # augment: apply the defect mutiple times
                            # augment apply the defect mutiple times
                        defect_batches = torch.split(sample_batched_defect['defect_image'].clone(), batch_size)
                        defect_batches_masks = torch.split(sample_batched_defect['defect_image_mask'].clone(), batch_size)
                        for defects_round in range(number_of_defect_per_image):
                            # print(defects_round)
                            if defects_round == 0:
                                input = [sample_batched_img_level['image'].clone(), sample_batched_img_level['img_mask'].clone(), defect_batches[defects_round], defect_batches_masks[defects_round], sample_batched_img_level['defect_location_mask'].clone() ]
                            else:
                                input = [output_photometric[0], output_photometric[1], defect_batches[defects_round], defect_batches_masks[defects_round], sample_batched_img_level['defect_location_mask'].clone() ]
                            # with torch.no_grad(): if not learning the policy, then we can keep this with no grad
                            output_photometric = policy_photometric(input) # output_photometric is [ Tensor: augmented image, Tensor: Defect mask]
                            augmented_image_photometric =  output_photometric[0].to(device)
                            defect_mask_photometric = output_photometric[1][:,:1,:,:].to(device) # the output_photometric mask is C x H x W
                        # forwrad
                        pred_defect_mask_photometric = model(augmented_image_photometric.to(device))
                        gt_defect_mask_photometric = defect_mask_photometric.to(device)[:,:1,:,:]
                        tr_loss_defect_photometric = (1-dice_weight)*bce(pred_defect_mask_photometric, gt_defect_mask_photometric) +  dice_weight*dice(pred_defect_mask_photometric, gt_defect_mask_photometric)

                        ###### calculate loss for augmented images tr_loss_defect_shear
                        # augment: apply the defect mutiple times
                            # augment apply the defect mutiple times
                        defect_batches = torch.split(sample_batched_defect['defect_image'].clone(), batch_size)
                        defect_batches_masks = torch.split(sample_batched_defect['defect_image_mask'].clone(), batch_size)
                        for defects_round in range(number_of_defect_per_image):
                            # print(defects_round)
                            if defects_round == 0:
                                input = [sample_batched_img_level['image'].clone(), sample_batched_img_level['img_mask'].clone(), defect_batches[defects_round], defect_batches_masks[defects_round], sample_batched_img_level['defect_location_mask'].clone() ]
                            else:
                                input = [output_shear[0], output_shear[1], defect_batches[defects_round], defect_batches_masks[defects_round], sample_batched_img_level['defect_location_mask'].clone() ]
                            # with torch.no_grad(): if not learning the policy, then we can keep this with no grad
                            output_shear = policy_shear(input) # output_shear is [ Tensor: augmented image, Tensor: Defect mask]
                            augmented_image_shear =  output_shear[0].to(device)
                            defect_mask_shear = output_shear[1][:,:1,:,:].to(device) # the output_shear mask is C x H x W
                        # forwrad
                        pred_defect_mask_shear = model(augmented_image_shear.to(device))
                        gt_defect_mask_shear = defect_mask_shear.to(device)[:,:1,:,:]
                        tr_loss_defect_shear = (1-dice_weight)*bce(pred_defect_mask_shear, gt_defect_mask_shear) +  dice_weight*dice(pred_defect_mask_shear, gt_defect_mask_shear)

                        ###### calculate loss for augmented images tr_loss_defect_rotate
                        # augment: apply the defect mutiple times
                            # augment apply the defect mutiple times
                        defect_batches = torch.split(sample_batched_defect['defect_image'].clone(), batch_size)
                        defect_batches_masks = torch.split(sample_batched_defect['defect_image_mask'].clone(), batch_size)
                        for defects_round in range(number_of_defect_per_image):
                            # print(defects_round)
                            if defects_round == 0:
                                input = [sample_batched_img_level['image'].clone(), sample_batched_img_level['img_mask'].clone(), defect_batches[defects_round], defect_batches_masks[defects_round], sample_batched_img_level['defect_location_mask'].clone() ]
                            else:
                                input = [output_rotate[0], output_rotate[1], defect_batches[defects_round], defect_batches_masks[defects_round], sample_batched_img_level['defect_location_mask'].clone() ]
                            # with torch.no_grad(): if not learning the policy, then we can keep this with no grad
                            output_rotate = policy_rotate(input) # output_rotate is [ Tensor: augmented image, Tensor: Defect mask]
                            augmented_image_rotate =  output_rotate[0].to(device)
                            defect_mask_rotate = output_rotate[1][:,:1,:,:].to(device) # the output_rotate mask is C x H x W
                        # forwrad
                        pred_defect_mask_rotate = model(augmented_image_rotate.to(device))
                        gt_defect_mask_rotate = defect_mask_rotate.to(device)[:,:1,:,:]
                        tr_loss_defect_rotate = (1-dice_weight)*bce(pred_defect_mask_rotate, gt_defect_mask_rotate) +  dice_weight*dice(pred_defect_mask_rotate, gt_defect_mask_rotate)

                        ###### calculate loss for augmented images tr_loss_defect_scale
                        # augment: apply the defect mutiple times
                            # augment apply the defect mutiple times
                        defect_batches = torch.split(sample_batched_defect['defect_image'].clone(), batch_size)
                        defect_batches_masks = torch.split(sample_batched_defect['defect_image_mask'].clone(), batch_size)
                        for defects_round in range(number_of_defect_per_image):
                            # print(defects_round)
                            if defects_round == 0:
                                input = [sample_batched_img_level['image'].clone(), sample_batched_img_level['img_mask'].clone(), defect_batches[defects_round], defect_batches_masks[defects_round], sample_batched_img_level['defect_location_mask'].clone() ]
                            else:
                                input = [output_scale[0], output_scale[1], defect_batches[defects_round], defect_batches_masks[defects_round], sample_batched_img_level['defect_location_mask'].clone() ]
                            # with torch.no_grad(): if not learning the policy, then we can keep this with no grad
                            output_scale = policy_scale(input) # output_scale is [ Tensor: augmented image, Tensor: Defect mask]
                            augmented_image_scale =  output_scale[0].to(device)
                            defect_mask_scale = output_scale[1][:,:1,:,:].to(device) # the output_scale mask is C x H x W
                        # forwrad
                        pred_defect_mask_scale = model(augmented_image_scale.to(device))
                        gt_defect_mask_scale = defect_mask_scale.to(device)[:,:1,:,:]
                        tr_loss_defect_scale = (1-dice_weight)*bce(pred_defect_mask_scale, gt_defect_mask_scale) +  dice_weight*dice(pred_defect_mask_scale, gt_defect_mask_scale)

                    ###### calculate loss for good augmented images tr_loss_img_level
                        pred_mask_img_good = model(sample_batched_good['image'].to(device))
                        gt_defect_mask_img_good = sample_batched_good['img_mask'].to(device)[:,:1,:,:]
                        tr_loss_img_good = (1-dice_weight)*bce(pred_mask_img_good, gt_defect_mask_img_good) +  (dice_weight)*dice(pred_mask_img_good, gt_defect_mask_img_good)# input mask is 3 channels, convert to 


                    ###### calculate loss for image level augmented images tr_loss_img_level
                        augmented_img_level_image, augmented_img_level_image_mask = image_level_aug_policy(sample_batched_img_level['image'].clone(), sample_batched_img_level['img_mask']) 
                        pred_mask_img_level = model(augmented_img_level_image.to(device))
                        gt_defect_mask_img_level = augmented_img_level_image_mask.to(device)[:,:1,:,:]
                        tr_loss_img_level = (1-dice_weight)*bce(pred_mask_img_level, gt_defect_mask_img_level) +  (dice_weight)*dice(pred_mask_img_level, gt_defect_mask_img_level)# input mask is 3 channels, convert to 

                    ###### calculate total loss
                        if optimize_weights:
                            tr_loss = weight_image_level_aug*tr_loss_img_level + lam_defect_cut_paste_photometirc*tr_loss_defect_photometric + lam_defect_cut_paste_rotate*tr_loss_defect_rotate + lam_defect_cut_paste_shear*tr_loss_defect_shear + lam_defect_cut_paste_scale*tr_loss_defect_scale + lam_good*tr_loss_img_good
                        else:
                            tr_loss = weight_image_level_aug*tr_loss_img_level + weight_defect_cut_paste_photometirc*tr_loss_defect_photometric + weight_defect_cut_paste_rotate*tr_loss_defect_rotate + weight_defect_cut_paste_shear*tr_loss_defect_shear + weight_defect_cut_paste_scale*tr_loss_defect_scale + weight_good*tr_loss_img_good

                        optimizer1.zero_grad()
                        
                        zero_hypergrad(get_hyper_train)
                        all_param_grads = grad(tr_loss, trainable_params,retain_graph=True, create_graph=True)# create_graph == true is necessary for further deffertication of the "all_param_grads", aka, 'd_train_loss_d_w'.

                        d_train_loss_d_w += gather_flat_grad(all_param_grads)
                        if tr_batch>= num_tr_accumulate:
                            break
                    d_train_loss_d_w /= num_tr_accumulate
                        
                    # # calculate val gradient
                    # num_val_accumulate = 6
                    d_val_loss_d_theta, direct_grad = torch.zeros(num_weights).to(device), torch.zeros(num_hypers).to(device)
                    model.train(), model.zero_grad()
                    val_loss_running = 0
                    for j_batch, sample_val in enumerate(dataloader_val):
                        pred_mask = model(sample_val['defect_image'].to(device))
                        gt_defect_mask = sample_val['defect_image_mask'][:,:1,:,:].to(device)
                        val_loss_ = (1-dice_weight)*bce(pred_mask, gt_defect_mask) +  dice_weight*dice(pred_mask, gt_defect_mask)
                        val_loss_running += val_loss_.item()
                        val_grad = grad(val_loss_, trainable_params)
                        d_val_loss_d_theta += gather_flat_grad(val_grad)# notice that this only usees current batch tr_loss to approximate the dataset val_loss
                        # if j_batch>=num_val_accumulate:
                        #     break
                    d_val_loss_d_theta = d_val_loss_d_theta/len(dataloader_val)
                    val_loss_running = val_loss_running/len(dataloader_val)
                    print('val_loss_running', val_loss_running)
                    elementary_lr = 1e-3
                    for param_group in optimizer1.param_groups:
                        elementary_lr = param_group['lr']
                        break

                    preconditioner = neumann_hyperstep_preconditioner(d_val_loss_d_theta, d_train_loss_d_w, elementary_lr, 1, trainable_params)/elementary_lr
                    # preconditioner = d_val_loss_d_theta.detach()
                    # print('d_val_loss_d_theta',d_val_loss_d_theta)
                    # print('d_train_loss_d_w', d_train_loss_d_w)
                    # print('preconditioner',preconditioner)

                    # for param in get_hyper_train():
                    #     print(param.grad)
                    indirect_grad = gather_flat_grad(
                        grad(d_train_loss_d_w, get_hyper_train(),grad_outputs=preconditioner.view(-1), retain_graph=True))

                    hypergrad = indirect_grad + direct_grad
                    hyper_grad += hypergrad
                hyper_grad /= num_hyper_grad_accumulate
                print('hyper_grad:', hyper_grad)

                zero_hypergrad(get_hyper_train)
                store_hypergrad(get_hyper_train, -hyper_grad) # gradient update
                hyper_optimizer.step()
                clip_hyper_train(get_hyper_train)
                print(30*'-')
                print('hyper_parameters: ', get_hyper_train())
                print(30*'-')

    # print(train_loss)
    if ep%10 == 0:
        torch.save(model, model_root+'model'+str(ep))

    train_loss_image_level = train_loss_image_level/len(dataloader_bkg)
    train_loss_dice = train_loss_dice/len(dataloader_bkg)
    train_loss_bce = train_loss_bce/len(dataloader_bkg)
    train_per_image_iou = train_per_image_iou/len(dataloader_bkg)
    train_loss.append([train_loss_bce, train_loss_dice, train_per_image_iou, train_loss_image_level, train_loss_defect1, train_loss_defect2, train_loss_defect3, train_loss_defect4])
    scheduler.step()
    hyper_scheduler.step()

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


# res = {}
# res['train_loss'] = train_loss
# res['val_loss'] = val_loss
# res['test_loss'] = test_loss
# res['val_loss_cat'] = val_loss_per_catg
# res['test_loss_cat'] = test_loss_per_catg

# # save results
# import json
# os.makedirs(store_root+'quant_resut/', exist_ok= True)
# with open(store_root+'quant_resut/'+'_rnd_seed_'+str(rnd_seed)+'_btsz_'+str(batch_size)+'_epcs_'+str(num_epochs)+'_lr_'+str(lr)+'_l1_'+ str(lam_defect1_cut_paste.item())+'_l2_'+ str(lam_defect2_cut_paste.item())+'_l3_'+ str(lam_defect3_cut_paste.item())+'_l4_'+ str(lam_defect4_cut_paste.item())+'.json', 'w') as f:
#     json.dump(res, f)

# final_dic = {}
# # final_dic["target"] = val_per_image_iou
# final_dic["policy_params"] = {}
# name_ls = ["br_l", "br_u", "cst_l", "cst_u", "sat_l", "sat_u", "hue_l", "hue_u", "d", "s", "scl_u"]
# value_count = 0
# for para_name in name_ls:
#     final_dic["policy_params"][para_name] = args.aug_paras_float[value_count]
#     value_count = value_count + 1
# final_dic["weighting_parameters"] = []
# final_dic["weighting_parameters"] = [lam_defect1_cut_paste.item(),lam_defect2_cut_paste.item(), 
# lam_defect3_cut_paste.item(), 
# lam_defect4_cut_paste.item() ]

# final_dic["train"] = train_per_image_iou
# final_dic["val"] = val_per_image_iou
# final_dic["test"] = test_per_image_iou

# with open(store_root+'quant_resut/'+'cut_paste_result'+str(rnd_seed)+'.json', 'a') as f_:
#     json.dump(final_dic, f_)

# os.makedirs(store_root+'model/', exist_ok= True)
# model_saving_path = store_root+'model/'+'cut_paste_mdl'+'_rnd_seed_'+str(rnd_seed)+'_batch_size_'+str(batch_size)+'_num_epochs_'+str(num_epochs)+'_lr_'+str(lr) +'_lam1_'+ str(lam_defect1_cut_paste.item())+'_lam2_'+ str(lam_defect2_cut_paste.item())+'_lam3_'+ str(lam_defect3_cut_paste.item())+'_lam4_'+ str(lam_defect4_cut_paste.item())+'.pth'

# torch.save(model, model_saving_path)