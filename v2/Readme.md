# Cut-paste package v2
This is the developed "cut-paste" package for online synthetic defect data generation. We tested it on the MVTech carpet data set (see the dataset part)
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
where 'gt' stores the binary segmentation masks and 'good' refers to clean background images. 

## Usage

Please first install related packages by running
```
pip install -r requirement.txt
```

This is v2 version of the cut-paste with bi-level learning capability. Please see RUN_v2.sbatch for example code to run.






