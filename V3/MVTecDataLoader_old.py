import numpy as np
import torch
import cv2
from torch.utils.data import Dataset
import os
import torchvision
import kornia as K


# MVtechDataset_bkg loader
class MVtechDataset_bkg(Dataset):
    """MVtech dataset."""

    def __init__(self, root_dir, dim = (256, 256), mask_dir = None, defect_ocation =None):
        """
        Arguments:
            mask_dir (string): Path to annotation mask file. # optional
            root_dir (string): Directory with all the images.
            defect_location (string): Path to defect location file. # optional
        """
        self.root_dir = root_dir
        self.img_names = sorted(os.listdir(self.root_dir))
        self.mask_dir = mask_dir
        self.defect_ocation = defect_ocation
        self.dim = dim
        if self.mask_dir:
            self.img_mask_names = sorted(os.listdir(self.mask_dir))
        if self.defect_ocation:
            self.defect_location_mask_names  = sorted(os.listdir(self.defect_ocation))


    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = self.img_names[idx]

        img: np.array = cv2.imread(self.root_dir + '/' + img_name)
        img: np.array = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img: np.array = cv2.resize(img, self.dim)
        img: torch.Tensor = (K.image_to_tensor(img).float())/255.

        if self.mask_dir:
            # img_mask_name = self.img_mask_names[idx]
        # read defect image mask
            img_mask_name = self.img_names[idx][:-4]+'_mask.png' 
            if os.path.isfile(self.mask_dir + "/" + img_mask_name):
                img_mask: np.array = cv2.imread(self.mask_dir + "/" + img_mask_name)
                img_mask: np.array = cv2.cvtColor(img_mask, cv2.COLOR_BGR2RGB)
                img_mask: np.array = cv2.resize(img_mask, self.dim)
                img_mask: torch.Tensor = (K.image_to_tensor(img_mask).float())/255. 
            else:# if there is no defect mask, this means the image is clean, we generate a zero a dfeect mask
                img_mask = torch.zeros_like(img)
            # img_mask: np.array = cv2.imread(self.mask_dir + '/' + img_mask_name)
            # img_mask: np.array = cv2.cvtColor(img_mask, cv2.COLOR_BGR2RGB)
            # img_mask: np.array = cv2.resize(img_mask, self.dim)
            # img_mask: torch.Tensor = (K.image_to_tensor(img_mask).float())/255.  
        else:
            img_mask = torch.zeros_like(img) # no defect on the original image

        if self.defect_ocation: 
            defect_location_mask_name = self.defect_location_mask_names[idx]
            defect_location_mask: np.array = cv2.imread(self.defect_ocation +'/' + defect_location_mask_name)
            defect_location_mask: np.array = cv2.cvtColor(defect_location_mask, cv2.COLOR_BGR2RGB)
            defect_location_mask: np.array = cv2.resize(defect_location_mask, self.dim)
            defect_location_mask: torch.Tensor = (K.image_to_tensor(defect_location_mask).float())/255. 
        else:
            defect_location_mask = torch.ones_like(img)  # possible to put defect everywhere  
        # # test
        # loc = int(self.dim[0]/4)
        # defect_location_mask = defect_location_mask * 0.0
        # defect_location_mask[:,loc: self.dim[0] - loc, loc: self.dim[0]-loc] = 1.0

        sample = {'image': img, 'img_mask': img_mask, 'defect_location_mask': defect_location_mask}
        return sample
    
# MVtechDataset_defect loader
class MVtechDataset_defect(Dataset):
    """MVtech dataset."""

    def __init__(self, root_dir, mask_dir, dim = (256, 256), crop = False):
        """
        Arguments:
            mask_dir (string): Path to annotation mask file. # optional
            root_dir (string): Directory with all the images.
        """
        self.root_dir = root_dir
        self.img_names = sorted(os.listdir(self.root_dir))
        self.mask_dir = mask_dir
        self.dim = dim
        self.crop = crop
        if self.mask_dir:
            self.img_mask_names = sorted(os.listdir(self.mask_dir))


    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = self.img_names[idx]
        # read defect image
        img: np.array = cv2.imread(self.root_dir + '/' + img_name)
        img: np.array = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img: np.array = cv2.resize(img, self.dim)
        img: torch.Tensor = (K.image_to_tensor(img).float())/255.

        
         
        # read defect image mask
        img_mask_name = self.img_names[idx][:-4]+'_mask.png' 
        # print(img_mask_name)
        if os.path.isfile(self.mask_dir + "/" + img_mask_name):
            img_mask: np.array = cv2.imread(self.mask_dir + "/" + img_mask_name)
            img_mask: np.array = cv2.cvtColor(img_mask, cv2.COLOR_BGR2RGB)
            img_mask: np.array = cv2.resize(img_mask, self.dim)
            img_mask: torch.Tensor = (K.image_to_tensor(img_mask).float())/255. 
        else: 
            print('mask and image mismatch')

        if self.crop:
            # crop mask and image then move to center (added 5 pixels offset)
            bbox = torchvision.ops.masks_to_boxes(img_mask[:1,:,:]) #of size [[x1, y1, x2, y2]]
            [[x1, y1, x2, y2]] =  bbox

            img_mask = img_mask[:, int(y1-10):int(y2+10), int(x1-10):int(x2+10)]
            img_mask = torchvision.transforms.CenterCrop(self.dim)(img_mask)

            img = img[:, int(y1-10):int(y2+10), int(x1-10):int(x2+10)]
            img = torchvision.transforms.CenterCrop(self.dim)(img)


        sample = {'defect_image': img, 'defect_image_mask': img_mask}
        return sample
