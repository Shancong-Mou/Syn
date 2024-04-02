# define augmentaion policy: copy and paste one defect onto one background
 
from typing import Tuple
import torch.nn as nn
from torch import Tensor
import torch
import random
from composition import PoissonCompositionLayer 
import kornia as K
import matplotlib.pyplot as plt

class SubPolicyStage(nn.Module):
    # this function defines the operation of a single subpolicy stage:
    # get one batch of defects and masks, copy and paste onto one batch of backgrounds
    # Each background will have only one defect samples
    # self.operations are a list of all operations 
    def __init__(self,
                operations: nn.ModuleList,
                apply_operations = True,
                apply_poisson = True
                 ):
        super(SubPolicyStage, self).__init__()
        self.operations = operations # a series of augmentations from kornia
        self.apply_operations = apply_operations # wether apply augmentation operations to defects
        self.apply_poisson = apply_poisson # whether apply poisson imaeg composition
        

    def forward(self,
                input: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor], # [bkg, bkg defect mask, defect, defect mask, possible defect locations],
                ) -> Tuple[Tensor, Tensor]:
        bkg, bkg_mask, defect, defect_mask, defect_location_masks = input
        batch_size = bkg.shape[0]
        # print(batch_size)
        # print(defect.shape[0])


        
        # here we do not learn the probability but implement an approximation
        # return (torch.stack([op(input) for op in self.operations]) * self.weights.view(-1, 1, 1, 1, 1)).sum(0)
        # defect augmentation:

        # Randomly apply augmentation to 50% of the samples
        n_aug_samples = max(1, (batch_size // 2))

        # shuffle the order of augmentations
        random.shuffle(self.operations)

        if self.apply_operations:
            for op in self.operations:
                # for each operation, apply it on half of batch of the images, in each iteration random shuffle the operations to mitigate the effect of augmentation orders
                sample_ids = torch.randperm(
                    n=batch_size, dtype=torch.long
                )[:n_aug_samples]
                # print(op)
                defect_aug = torch.index_select(defect, dim=0, index=sample_ids)
                defect_mask_aug = torch.index_select(defect_mask, dim=0, index=sample_ids)

                # apply the same operation on selection defects
                defect_aug = op(defect_aug)

                # apply the same operation on the mask if it is geometric 
                if op.is_geometric == True:
                    defect_mask_aug = op(defect_mask_aug, params=op._params)
                else:
                    defect_mask_aug = defect_mask_aug
                
                # copy augmented samples to tensor
                defect = torch.index_copy(defect, dim=0, source=defect_aug, index=sample_ids)
                defect_mask = torch.index_copy(defect_mask, dim=0, source=defect_mask_aug, index=sample_ids)
                # for i in defect_mask:
                #     plt.imshow(i[0,:,:].detach().numpy())
                #     plt.show()
        # Apply translation operation on bthces of defects, according to the translation mask: defect_location_batch
        # sample a location from defect_location_masks
        _defect_translation_coordinates = [] # batch
        
        for defect_location_mask in defect_location_masks:
            _defect_location_mask = defect_location_mask[0,:,:] # WxH
            # plt.imshow(defect_location_mask[0,:,:])
            # plt.legend()
            # plt.show()
            _locations = (_defect_location_mask >= 1e-5).nonzero(as_tuple=False).float()# extract locations of non_zeros
            _x_center, _y_center = _defect_location_mask.shape[-2:]
            x_center, y_center = (_x_center/2), (_y_center/2)# find center of the image (also the defect center)
            # random slect a locations to put defect; TODO experiment design for defect location 
            indices = random.sample(range(len(_locations)),1) # 1 x 2
            _translate_dist = _locations[indices] - torch.tensor([x_center, y_center]) # 1 x 2
            translate_dist = torch.tensor([_translate_dist[0,1],_translate_dist[0,0]])
            translate_dist = torch.unsqueeze(translate_dist, 0)
            _translate_coordinates = [] # translation coordinates to be feed into translate_xy
            for dist in translate_dist:
                _translate_coordinates.append(torch.tensor(dist[None, :])) #  1x2
            translate_coordinates = torch.stack(_translate_coordinates) # 1 x 1 x 2

            _defect_translation_coordinates.append(translate_coordinates) # B x 1 x 1 x 2
        defect_translation_coordinates = torch.stack(_defect_translation_coordinates) # B x 1 x 1 x 2
        # print(defect_translation_coordinates)
        
        # divide to each batch of defects
        defect_location_batch = torch.squeeze(defect_translation_coordinates)
        # print(defect_location_batch.shape)
        # print(defect_location_batch)
        defect_new_loc = self.translate_xy(defect, defect_location_batch)
        defect_mask_new_loc = self.translate_xy(defect_mask, defect_location_batch)


        # Apply poisson image composition to cut and past defect to backgrounds
        if self.apply_poisson:
            poisson_layer = PoissonCompositionLayer()
            img_aug = poisson_layer(bkg, defect_new_loc, defect_mask_new_loc) # possion image composition 
        else:
            img_aug = bkg * ( 1 - defect_mask_new_loc ) + defect_new_loc * defect_mask_new_loc

        # add orginal defect mask 
        defect_mask_new_loc = defect_mask_new_loc + bkg_mask


        
        return img_aug, defect_mask_new_loc.clip( min=0, max=1)
    

    def translate_xy(self, img: torch.Tensor,
            mag: torch.Tensor) -> torch.Tensor:
    # mag = torch.stack([mag, torch.zeros_like(mag)], dim=1)
        return K.geometry.transform.translate(img, mag)