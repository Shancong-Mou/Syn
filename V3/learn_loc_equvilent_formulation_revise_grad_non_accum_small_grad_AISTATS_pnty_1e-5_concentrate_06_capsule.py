# In this script, we try to optimzie the weight of each augmentations 
# \sum_j \lambda_j \sum_{i_j} L(x_{i_j},y_{i_j})
# Where \lambda 1-5:
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
import matplotlib.pyplot as plt
from torch import Tensor
from typing import Tuple
from diffAugs import _Jitter, _Rotation, _Shear, _Scale
from MVTecDataLoader_old import MVtechDataset_bkg, MVtechDataset_defect
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from AugmentPolicy_take_location_input_06 import SubPolicyStage
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


num_epochs = 150 # number of epochs
root = "./Defect_aug_with_default_policy_add_clean_only_parameter_MVTech_start30_capsule_random_location/" # root to store
data_root = 'capsule' # root to data
dice_weight = 0.5 # loss metric: dice or bce
lr = 0.000125 # learning rate
rnd_seed = 0 # random seed
batch_size = 2 #batch size
lr_schedule_step_size = 20

br_l_, br_u_, cst_l_, cst_u_, sat_l_, sat_u_, hue_l_, hue_u_, d_, s_, scl_u_ = [0.1, 1.9, 0.1, 1.9, 0.1, 1.9, 0.0, 0.5, 30.0, 0.3, 2.0]
train_from_scrach = False
warm_up = 0
weight_image_level_aug, weight_defect_cut_paste = [1.0, 1.0]
optimize_cut_paste_policy_parameter = False
optimize_image_level_aug_policy_parameter = False
optimize_weights = False
number_of_defect_per_image = 1
apply_augmentations = True
apply_poisson = False
save_all_images = True
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
    return elementary_lr * preconditioner
    # return preconditioner


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
    # current_index = 0
    # for index, p in enumerate(get_hyper_train()):
    #     p.grad = get_hyper_train_grad[index]
    current_index = 0
    for p in get_hyper_train():
        p_num_params = np.prod(p.shape)
        p.grad = get_hyper_train_grad[current_index:current_index + p_num_params].view(p.shape)
        current_index += p_num_params

# specify iamge storgation
# setup storage    
store_root = root #'./results/Whole_image_aug/'
os.makedirs(store_root, exist_ok= True)

image_root = store_root+'_rnd_seed_'+str(rnd_seed)+'_batch_size_'+str(batch_size)+'_num_epochs_'+str(num_epochs)+'_lr_'+str(lr)+ '_lr_schedule_step_size_'+str(lr_schedule_step_size) + 'loc_map_equvlent_non_accum_small_grad_AISTATS_l1_1e-5_concentration_06/'
os.makedirs(image_root, exist_ok= True)
model_root = store_root+'_rnd_seed_'+str(rnd_seed)+'_batch_size_'+str(batch_size)+'_num_epochs_'+str(num_epochs)+'_lr_'+str(lr)+ '_lr_schedule_step_size_'+str(lr_schedule_step_size) + 'model_l1_1e-5_concentration_06/'
os.makedirs(model_root, exist_ok= True)

## import Unet
from torch.utils.data import DataLoader
# import a Unet backbone with imagenet pretrained weights
import segmentation_models_pytorch as smp
model_saving_path = './Defect_aug_with_default_policy_add_clean_only_parameter_MVTech_start30_capsule_random_location_1defct_per_image/_rnd_seed_'+str(rnd_seed)+'_batch_size_2_num_epochs_150_lr_0.00025_lr_schedule_step_size_20model/model30'

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

# map_pred_net = AutoEncoderCNN()
map_pred_net = smp.Unet(
    encoder_name="resnet18",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=1,                      # model output channels (number of classes in your dataset)
    activation = 'sigmoid'
)
model.to(device)
map_pred_net.to(device)

from segmentation_models_pytorch.losses import DiceLoss
from torch import optim



# define batch size and numbe of defects per iamge
number_of_defect_per_image = 1 
dim = (256,256) # image shape to be resized


# load data
# load background dataset # use additional clean background
MVtechDataset_good_dataset = MVtechDataset_bkg(root_dir= "./data/" + data_root +"/good",dim =dim)
dataloader_good = DataLoader(MVtechDataset_good_dataset, batch_size=batch_size,
                        shuffle=True, num_workers=0, drop_last= True)
# # load background dataset # use the clean image data set as background
# MVtechDataset_bkg_dataset = MVtechDataset_bkg(root_dir= "./data/" + data_root +"/good",dim =dim, defect_ocation = "./data/" + data_root +"/good_object_location_mask")
MVtechDataset_bkg_dataset = MVtechDataset_bkg(root_dir= "./data/" + data_root +"/good",dim =dim)

dataloader_bkg = DataLoader(MVtechDataset_bkg_dataset, batch_size=batch_size,
                        shuffle=True, num_workers=0, drop_last= True)

# load defect dataset (defects from the training set)
MVtechDataset_defect_dataset = MVtechDataset_defect(root_dir= "./data/" + data_root +"/combined/train", dim = dim,
                                                    mask_dir= "./data/" + data_root +"/combined/train_gt",
                                                    crop =True)
dataloader_defect = DataLoader(MVtechDataset_defect_dataset, batch_size=(batch_size)*number_of_defect_per_image,
                        shuffle=True, num_workers=0, drop_last= True)

# load defect label dataset (defects from the training set), but with defect structure only from background 
MVtechDataset_good_as_defect_dataset = MVtechDataset_defect(root_dir= "./data/" + data_root +"/train", dim = dim,
                                                    mask_dir= "./data/" + data_root +"/combined/train_gt",
                                                    crop =True)
dataloader_good_as_defect = DataLoader(MVtechDataset_good_as_defect_dataset, batch_size=(batch_size)*number_of_defect_per_image,
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
# ShearX = _Shear(shearX)
# ShearY = _Shear(shearY)
# Scale = _Scale(scale)
cut_paste_operations = [Jitter, Rotation]#, ShearX, ShearY]

policy = SubPolicyStage(operations = cut_paste_operations, apply_operations = apply_augmentations, apply_poisson = apply_poisson)


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




# lam_defect_cut_paste = _p(weight_defect_cut_paste)
# hyper_parameters = hyper_parameters + [lam_defect_cut_paste]

# map_pred_net_parameters = gather_flat_grad(map_pred_net.parameters())

map_pred_net_parameters = [p for p in map_pred_net.parameters()]
hyper_parameters = hyper_parameters + map_pred_net_parameters


def get_hyper_train():
    return hyper_parameters

def get_hyper_train_flat():
    return torch.cat([p.view(-1) for p in hyper_parameters])

from torch.optim.lr_scheduler import StepLR
# hyper_optimizer = optim.Adam(get_hyper_train(), lr = 1e-1)
hyper_optimizer = optim.Adam(hyper_parameters, lr = 1e-4)


hyper_scheduler =StepLR(hyper_optimizer, step_size=20, gamma=0.5)

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
    NUM_ACCUMULATION_STEPS = 1 # TODO change to 1
    model.train()
    iterator = iter(dataloader_defect)
    for i_batch, sample_batched_img_level in enumerate(tqdm(dataloader_bkg, desc = 'epoch' + str(ep) +'_train')): # use image_level_aug as main iteration
        sample_batched_defect = next(iterator, None)
        if sample_batched_defect == None:
            iterator = iter(dataloader_defect)
            sample_batched_defect = next(iterator, None)


        ###### calculate loss for augmented images tr_loss_defect
        # augment: apply the defect mutiple times
            # augment apply the defect mutiple times
        defect_batches = torch.split(sample_batched_defect['defect_image'].clone(), batch_size)
        defect_batches_masks = torch.split(sample_batched_defect['defect_image_mask'].clone(), batch_size)
        location_maps = map_pred_net(sample_batched_img_level['image'].to(device)) # batch sie == 2
        # print("location_maps", location_maps.shape)
        # print("sample_batched_img_level['defect_location_mask']", sample_batched_img_level['defect_location_mask'].shape)
        # break
        for defects_round in range(number_of_defect_per_image):
            # print(defects_round)
            if defects_round == 0:
                input = [sample_batched_img_level['image'].clone(), sample_batched_img_level['img_mask'].clone(), defect_batches[defects_round], defect_batches_masks[defects_round], location_maps.detach().clone().cpu() ]
            else:
                input = [output[0], output[1], defect_batches[defects_round], defect_batches_masks[defects_round], location_maps.detach().clone().cpu() ]
            # with torch.no_grad(): if not learning the policy, then we can keep this with no grad
            output = policy(input) # output is [ Tensor: augmented image, Tensor: Defect mask]
            augmented_image =  output[0].to(device)
            defect_mask = output[1][:,:1,:,:].to(device) # the output mask is C x H x W
        ## save location map image
        plt.matshow(torch.permute(location_maps[0], (1, 2, 0)).detach().cpu().numpy(), cmap='hot' )
        plt.title('Displaying image using Matplotlib')
        plt.colorbar()
        # plt.show(block=False)
        # plt.close()
        plt.savefig(image_root+'ep'+str(ep)+'ibtach'+str(i_batch)+'loc_map'+'.png', bbox_inches='tight')
        plt.close()
        
        aug_weight = (location_maps*defect_mask).sum((1,2,3), keepdim=True)/defect_mask.sum((1,2,3), keepdim=True).to(device)
        # forwrad
        pred_defect_mask = model(augmented_image.to(device))
        gt_defect_mask = defect_mask.to(device)
        tr_loss_defect = 0
        for index in range(pred_defect_mask.shape[0]):
            tr_loss_defect += aug_weight[index] * ( (1-dice_weight)*bce(pred_defect_mask[index], gt_defect_mask[index]) +  dice_weight*dice(pred_defect_mask[index], gt_defect_mask[index]) )


        ###### calculate loss for image level augmented images tr_loss_img_level
        augmented_img_level_image, augmented_img_level_image_mask = image_level_aug_policy(sample_batched_img_level['image'].clone(), sample_batched_img_level['img_mask']) 
        pred_mask_img_level = model(augmented_img_level_image.to(device))
        gt_defect_mask_img_level = augmented_img_level_image_mask.to(device)[:,:1,:,:]
        for index in range(pred_defect_mask.shape[0]):
            tr_loss_img_level = aug_weight[index]* ((1-dice_weight)*bce(pred_mask_img_level[index], gt_defect_mask_img_level[index]) +  (dice_weight)*dice(pred_mask_img_level[index], gt_defect_mask_img_level[index]))# input mask is 3 channels, convert to 

        ###### calculate total loss
        tr_loss = weight_image_level_aug*tr_loss_img_level + weight_defect_cut_paste*tr_loss_defect


        # optimize
        tr_loss /= NUM_ACCUMULATION_STEPS
         
        tr_loss.backward(retain_graph=True)
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

        if save_all_images:
            numimage = batch_size
            if ep%30==0:
                for num_image in range(numimage):
                    fig, axs = plt.subplots(4, 1, figsize=(8, 20))
                    axs[0].imshow(K.tensor_to_image(augmented_image[num_image]))
                    axs[0].set_title('train input image')
                    axs[1].imshow(K.tensor_to_image(pred_defect_mask[num_image]))
                    axs[1].set_title('train predicted mask')
                    thred_pred_mask = Prob_mask2SigmoidMask(pred_defect_mask[num_image])
                    axs[2].imshow(K.tensor_to_image(thred_pred_mask))
                    axs[2].set_title('threded predicted mask')
                    axs[3].imshow(K.tensor_to_image(gt_defect_mask[num_image]))
                    axs[3].set_title('train true mask')

                    plt.savefig(image_root+'ep'+str(ep)+'ibtach'+str(i_batch)+'train_seg_iter'+'img'+str(num_image)+'.png', bbox_inches='tight')
                    plt.close()

        # hyper_step
        # print(f"num_weights : {num_weights}, num_hypers : {num_hypers}")
        if ep>=warm_up:# and ep%5 == 0:
            if i_batch%1 == 0:
                # accumulate hypergrad
                num_hyper_grad_accumulate = 1
                hyper_grad = 0
                for hyper_step in range(num_hyper_grad_accumulate):
                    model.train(), model.zero_grad()
                    # calculate d_train_d_w
                    d_train_loss_d_w = gather_flat_grad(grad(tr_loss, trainable_params,retain_graph=True, create_graph=True))

                    # print("weight_image_level_aug", weight_image_level_aug)
                    # print('tr_loss_img_level', tr_loss_img_level)
                    # print('weight_defect_cut_paste',weight_defect_cut_paste)
                    # print('tr_loss_defect',tr_loss_defect)
                    # print('tr_loss',tr_loss)
                    # print("d_train_loss_d_w",d_train_loss_d_w)
                    # print("location_maps", location_maps)

                    # # calculate val gradient
                    # num_val_accumulate = 6
                    d_val_loss_d_theta, direct_grad = torch.zeros(num_weights).to(device), torch.zeros(num_hypers).to(device)
                    model.train(), model.zero_grad()
                    val_loss_running = 0
                    for j_batch, sample_val in enumerate(dataloader_val):
                        pred_mask = model(sample_val['defect_image'].to(device))
                        gt_defect_mask = sample_val['defect_image_mask'][:,:1,:,:].to(device)
                        location_maps = map_pred_net(sample_val['defect_image'].to(device))
                        val_loss_ = (1-dice_weight)*bce(pred_mask, gt_defect_mask) +  dice_weight*dice(pred_mask, gt_defect_mask) + 0.00001*nn.L1Loss()(location_maps,torch.zeros_like(location_maps)) ### the second term can be set to a very small number, you can experiment to find a good value here
                        #####
                        # print('val_loss_', val_loss_)
                        val_loss_running += val_loss_.item()
                        val_grad = grad(val_loss_, trainable_params)
                        d_val_loss_d_theta += gather_flat_grad(val_grad)# notice that this only usees current batch tr_loss to approximate the dataset val_loss
                        # direct_grad_ = grad(val_loss_, get_hyper_train())
                        direct_grad += gather_flat_grad(grad(val_loss_, get_hyper_train()))
                        # if j_batch>=num_val_accumulate:
                        #     break
                    d_val_loss_d_theta = d_val_loss_d_theta/len(dataloader_val)
                    direct_grad = direct_grad/len(dataloader_val)
                    val_loss_running = val_loss_running/len(dataloader_val)
                    elementary_lr = 1e-3
                    for param_group in optimizer1.param_groups:
                        elementary_lr = param_group['lr']
                        break

                    preconditioner = neumann_hyperstep_preconditioner(d_val_loss_d_theta, d_train_loss_d_w, elementary_lr, 3, trainable_params)
                    # preconditioner = d_val_loss_d_theta.detach()
                    # print('d_val_loss_d_theta',d_val_loss_d_theta)
                    # print('d_train_loss_d_w', d_train_loss_d_w)
                    # print('preconditioner',preconditioner)
                     
                    # for param in get_hyper_train():
                    #     print(param.grad)
                    indirect_grad = gather_flat_grad(
                        grad(d_train_loss_d_w, get_hyper_train(),grad_outputs=preconditioner.view(-1), retain_graph=True))
                    # print('indirect_grad:', indirect_grad.norm())
                    hypergrad = indirect_grad - direct_grad
                    hyper_grad += hypergrad
                hyper_grad /= num_hyper_grad_accumulate
                print('ep'+str(ep)+"_ibatch"+str(i_batch)+'_val_loss_running', val_loss_running)
                # print('hyper_grad:', hyper_grad.norm())
                # print('d_train_loss_d_w', d_train_loss_d_w.norm())
                # print("preconditioner", preconditioner.norm())

                zero_hypergrad(get_hyper_train)
                store_hypergrad(get_hyper_train, -hyper_grad) # gradient update

                hyper_optimizer.step()
                # clip_hyper_train(get_hyper_train)
                # print(30*'-')
                # print('hyper_parameters: ', get_hyper_train())
                # print(30*'-')

    # print(train_loss)
    if ep%10 == 0:
        torch.save(model, model_root+'model'+str(ep))

    train_loss_image_level = train_loss_image_level/len(dataloader_bkg)
    train_loss_dice = train_loss_dice/len(dataloader_bkg)
    train_loss_bce = train_loss_bce/len(dataloader_bkg)
    train_per_image_iou = train_per_image_iou/len(dataloader_bkg)
    train_loss.append([train_loss_bce, train_loss_dice, train_per_image_iou, train_loss_image_level, train_loss_defect1, train_loss_defect2, train_loss_defect3, train_loss_defect4])
    # if ep<=40:
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
        # save image
        if ep >=15:
            numimage = sample_val['defect_image'].shape[0]
            if save_all_images:
                for num_image in range(numimage):
                    fig, axs = plt.subplots(4, 1, figsize=(8, 20))
                    axs[0].imshow(K.tensor_to_image(sample_val['defect_image'][num_image]))
                    axs[0].set_title('Input image')
                    axs[1].imshow(K.tensor_to_image(pred_mask[num_image]))
                    axs[1].set_title('Predicted mask')
                    thred_pred_mask = Prob_mask2SigmoidMask(pred_mask[num_image])
                    axs[2].imshow(K.tensor_to_image(thred_pred_mask))
                    axs[2].set_title('Binary predicted mask')
                    axs[3].imshow(K.tensor_to_image(gt_defect_mask[num_image]))
                    axs[3].set_title('True mask')

                    plt.savefig(image_root+'ep'+str(ep)+'ibtach'+str(k_batch)+'val_seg_iter'+'img'+str(num_image)+'.png', bbox_inches='tight')
                    plt.close()

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
        if ep >=15:
            numimage = sample_test['defect_image'].shape[0]
            if save_all_images:
                for num_image in range(numimage):
                    fig, axs = plt.subplots(4, 1, figsize=(8, 20))
                    axs[0].imshow(K.tensor_to_image(sample_test['defect_image'][num_image]))
                    axs[0].set_title('Input image')
                    axs[1].imshow(K.tensor_to_image(pred_mask[num_image]))
                    axs[1].set_title('Predicted mask')
                    thred_pred_mask = Prob_mask2SigmoidMask(pred_mask[num_image])
                    axs[2].imshow(K.tensor_to_image(thred_pred_mask))
                    axs[2].set_title('Binary predicted mask')
                    axs[3].imshow(K.tensor_to_image(gt_defect_mask[num_image]))
                    axs[3].set_title('True mask')

                    plt.savefig(image_root+'ep'+str(ep)+'ibtach'+str(k_batch)+'test_seg_iter'+'img'+str(num_image)+'.png', bbox_inches='tight')
                    plt.close()
    test_loss_dice = test_loss_dice/len(dataloader_test)
    test_loss_bce = test_loss_bce/len(dataloader_test)
    test_per_image_iou = test_per_image_iou/len(dataloader_test)
    test_loss.append([test_loss_bce, test_loss_dice, test_per_image_iou])
          
    print("ep", ep, "train_loss", f"{train_loss_bce:.3}", f"{train_loss_dice:.3}", f"{train_per_image_iou:.3}", "val_loss", f"{val_loss_bce:.3}", f"{val_loss_dice:.3}",f"{val_per_image_iou:.3}", "test_loss", f"{test_loss_bce:.3}", f"{test_loss_dice:.3}", f"{test_per_image_iou:.3}", )

res = {}
res['train_loss'] = train_loss
res['val_loss'] = val_loss
res['test_loss'] = test_loss
print(res)
# print()


