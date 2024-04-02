# Cut-paste package v3
This is the developed "cut-paste" package for online synthetic defect data generation. Compared to v2, we added the important location learning funtionality. More detials can be found in our paper ```Synth4Seg - Learning Defect Data Synthesis for Defect Segmentation using Bi-level Optimization``` in submission.

Compared to v2, the updates are highlited using yellow ==color==.
## Modules
### **Differentiable defect augmentation module: ```diffAugs.py```**
This module supports 8 common augmentation operations in a differentiable way, including:
1. Geometric augmentation operations

    Rotation, shearX, shearY, Scale 

2. Photometric augmentation operations

    Brightness, Contrast, Saturation, Hue

which aims to augment the defect images.

### **Differentiable Poisson image composition module: ```composition.py```**
This module supports differntiable poisson image composition which aims for the seamless integration of augmented defect image with the target background images.

### **Differentiable augmentation policy module: ```AugmentPolicy.py```**
This module integrates ```diffAugs.py``` and ```composition.py``` such that it defines the overall augmentation policy, with the input of ```target background image``` and ```cropped defect from the library```, output the ```[bkg, bkg defect mask, defect, defect mask, possible defect locations]```:

```python
from AugmentPolicy import SubPolicyStage

policy = SubPolicyStage(operations = [list_of_augmentation_operations], apply_operations = [wether apply augmentation operrations], apply_poisson = [wether apply poisson image composition])

[synthetic_defect_image, synthetic_defect_image_annotation]  = policy([bkg, bkg defect mask, defect, defect mask, possible defect locations]) # more details can be found in the example code file

```
where ```possible defect locations``` is a binary mask indicating possible defect locations on the target background. By default, a matrix of 1's indicates defects can appear anywhere on the target image.

<mark> In the location optimziation script, we use the ```AugmentPolicy_take_location_input_06.py``` <mark>, where the main differnce is that it will take the current predicted mask and only paste defct onto locations greater than 0.6.
```python
_locations = (_defect_location_mask >= 0.6).nonzero(as_tuple=False).float()# extract locations of non_zeros 
```

### **Main module: ```main.py```** 
This is the main module in the developed package, which integrates the following parts:
1. Data loader: ```MVTecDataLoader_old.py``` which incldues the following two loaders: 

    - Background loader ```MVtechDataset_bkg()```
    - Defect loader ```MVtechDataset_defect()```

2. Data augmentation and composition module

3. Segmentation model training module (binary segmentation for now)

## Dataset structure
Please download the dataset from ```https://www.dropbox.com/t/uLVEYghgy0JAzCTR``` and put it in the root directory.

Use Carpet dataset as an example

```
./Data/Carpet
- Combined 
  -- test
  -- test_gt
  -- train
  -- train_gt
  -- val
  -- val_gt
- good
```
<mark>New dataset format<mark>
```
./Data/Carpet
- Combined 
  -- test
  -- test_gt
  -- train
  -- train_gt
  -- val
  -- val_gt
- good
- good_object_location_mask
```
where 'gt' stores the binary segmentation masks and 'good' refers to clean background images. 

## <mark>The learning location code<mark>
<mark>```learn_loc_equvilent_formulation_revise_grad_non_accum_small_grad_AISTATS_pnty_1e-5_concentrate_06_capsule.py```<mark>

The main difference in this version is the location map learning network:
```python
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
```
where the parameters in the network is the high-dimensional upper level optimziation variable. (Much higher dimension compared to the previous mixing ratio experiments.) 
## <mark>Usage<mark>
1. Please first install related packages by running
    ```
    pip install -r requirement.txt
    ```
2. Please downlaod necessary data and pretrained models checkpoints using the following link: https://www.dropbox.com/t/FVvxCDaT6BqxGTmR

    Please just unzip the file and put it to the current directory ```./v3/```.

3. This is v3 version of the cut-paste with bi-level learning capability. Please see ```RUN_cut_paste_bi_level_opt_optimzie_location_penalty.sbatch``` for example code to run, where 3 example codes are attached, including:
```python
# random location baseline (Cut&Paste - Random locations )
Defect_aug_with_default_policy_add_clean_augmentation_proportion_only_optimize_aug_policy_add_each_aug_MV_tech_to_background_optimize_location_baseline.py
# this is the object location baseline (Groundtruth product location )
Defect_aug_with_default_policy_add_clean_augmentation_proportion_only_optimize_aug_policy_add_each_aug_MV_tech_to_background_optimize_location_on_object.py
# optimziation location (Cut&Paste - Learned locations )
learn_loc_equvilent_formulation_revise_grad_non_accum_small_grad_AISTATS_pnty_1e-5_concentrate_06_capsule.py
```







