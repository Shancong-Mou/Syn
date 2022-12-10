import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import os
import numpy as np
from PIL import Image
import glob
import json
from matplotlib import pyplot as plt
import os
from tqdm import tqdm
import albumentations as A
import pandas as pd
import random

import scipy.sparse

# image composition
import cv2



def PoisImgComp(source, mask, loc, target, bbox):
    # Input: (this is an optimized version of possion image composition by Shancong, where OpenCV's implementation are not optimla and will destroy the defects)
    #   source: input source image, same size as target image
    #   mask: input source defective region seg mask (binary, 1 defecitve pixel, 0 o.w.)
    #   target: taget image to generate defect
    #   loc: location of the defect on atrget image
    #   bbox: bbox of source defect
    # Output:
    #   target_img: output synthetic defective image

    # move mask (x,y, x+w, y+h) to loc (new x,y location)
    x, y, w, h = bbox
    num_rows, num_cols = source.shape[:2] 
    translation_matrix = np.float32([ [1,0,loc[0]-x], [0,1,loc[1]-y] ])
    source = cv2.warpAffine(source, translation_matrix, (num_cols, num_rows)) 

    mask =   cv2.warpAffine(mask, translation_matrix, (num_cols, num_rows)) 

    # accelerate computation (only compose defect region plus minus 10 pixels)
    target_img = target.copy()[ loc[1]-10:loc[1]+h+10, loc[0]-10:loc[0]+w+10]
    source_img = source.copy()[ loc[1]-10:loc[1]+h+10, loc[0]-10:loc[0]+w+10]
    mask_img = mask.copy()[ loc[1]-10:loc[1]+h+10, loc[0]-10:loc[0]+w+10]
    def laplacian_matrix(n, m):   
        mat_D = scipy.sparse.lil_matrix((m, m))
        mat_D.setdiag(-1, -1)
        mat_D.setdiag(4)
        mat_D.setdiag(-1, 1)
            
        mat_A = scipy.sparse.block_diag([mat_D] * n).tolil()
        
        mat_A.setdiag(-1, 1*m)
        mat_A.setdiag(-1, -1*m)
        
        return mat_A

    y_max, x_max = target_img.shape[:-1]
    y_min, x_min = 0, 0
    x_range = x_max - x_min
    y_range = y_max - y_min

    mask_img[mask_img != 0] = 1

    mat_A = laplacian_matrix(y_range, x_range)
    laplacian = mat_A.tocsc()

    for y in range(1, y_range - 1):
        for x in range(1, x_range - 1):
            if mask_img[y, x] == 0:
                k = x + y * x_range
                mat_A[k, k] = 1
                mat_A[k, k + 1] = 0
                mat_A[k, k - 1] = 0
                mat_A[k, k + x_range] = 0
                mat_A[k, k - x_range] = 0
    mat_A = mat_A.tocsc()

    from scipy.sparse.linalg import spsolve

    mask_flat = mask_img.flatten()    
    for channel in range(source_img.shape[2]):
        source_flat = source_img[y_min:y_max, x_min:x_max, channel].flatten()
        target_flat = target_img[y_min:y_max, x_min:x_max, channel].flatten()        

        # inside the mask:
        # \Delta f = div v = \Delta g       
        alpha = 1
        mat_b = laplacian.dot(source_flat)*alpha

        # outside the mask:
        # f = t
        mat_b[mask_flat == 0] = target_flat[mask_flat == 0]
        
        x = spsolve(mat_A, mat_b)    
        x = x.reshape((y_range, x_range))
        x[x > 255] = 255
        x[x < 0] = 0
        x = x.astype('uint8')
        
        target_img[y_min:y_max, x_min:x_max, channel] = x
    
    target_cmpst = target.copy()
    target_cmpst[ loc[1]-10:loc[1]+h+10, loc[0]-10:loc[0]+w+10] = target_img
    return target_cmpst


# This function is a preparation function, which will extract and move the defct region
# from the source image to the center of the image. 
# It will generate a new dataset of those defects, as well as an annotation file names:
# 'training_defect_library_annotations.json'
# We conduct this step to facilitate the downstream image augmentation operation
def defect_source_prep(image_path, images_names_train, annotations, out_put_dir):
    # image_path: directory of input images
    # images_names_train: list of defect image names 
    # annotations: annotations correponding to the images names
    # out_put_dir: generated defect source dataset

    # crop defects from images_names_train and move to center of the image and save
    count = 0
    defect_annotations = {}
    os.makedirs(out_put_dir,exist_ok= True)
    for image_name in tqdm(sorted(images_names_train), desc="preparing_defect_images", position=0):
        # image = Image.open(os.path.join(image_path, image_name)).convert('RGB')
        image = cv2.imread(os.path.join(image_path, image_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[0], image.shape[1]
        defects = annotations[image_name]
        for defect in tqdm(defects, desc=" defects", position=1, leave=False):
            try: # some annotation maynot have segmentation annotation, just skip
                individual_defect_annotations = {}
                # get_defect_parameters
                x, y, w, h = defect["bbox"]
                centering_dic = {}
                centering_dic['x'] = -int(x+w/2-width/2)
                centering_dic['y'] = -int(y+h/2-height/2)
                bbox =  [defect["bbox"]]
                seg1 = [poly for poly in defect['seg1']]
                class_labels = [defect['category']]
                # define transform
                transform_image = A.Compose(
                [A.Affine(translate_px = centering_dic, p=1)],
                bbox_params=A.BboxParams(format="coco", label_fields=["class_labels"]),
                )
                # transform (move to center)
                transformed_img = transform_image(
                    image=image,
                    bboxes=bbox,
                    class_labels=class_labels
                    )
                    
                transformed_image = transformed_img["image"]
                transformed_bboxes = transformed_img["bboxes"][0]

                segs_after_transform = []
                for seg in seg1:
                    transform_seg = A.Compose(
                    [A.Affine(translate_px = centering_dic, p=1)],
                    keypoint_params=A.KeypointParams(format="xy"),
                    )
                    transformed_seg = transform_seg(
                        image=image,
                        keypoints=seg
                        )                
                    transformed_keypoints = transformed_seg["keypoints"]
                    segs_after_transform.append([[int(pt) for pt in poly] for poly in transformed_keypoints])

                bbox = [ int(pt) for pt in transformed_bboxes]
                catgry = defect['category']
                
                individual_defect_annotations['bbox'] = bbox
                individual_defect_annotations['seg1'] = segs_after_transform
                individual_defect_annotations['category'] = catgry
                defect_annotations[str(count)] = [individual_defect_annotations]
                Image.fromarray(transformed_image).save(out_put_dir+str(count)+'.png')
                count = count + 1

                # # disply raw images
                # x, y, w, h = defect["bbox"]
                # p1, p2 = (int(x), int(y)), (int(x+w), int(y+h))
                # mask = np.zeros((image.shape[0], image.shape[1], 3)).astype(np.uint8)
                # mask[:, :, 1] = cv2.rectangle(mask[:, :, 1].copy(), p1, p2, color=255, thickness=4)
                # seg1 = [np.array(poly) for poly in defect['seg1']]
                # mask[:, :, 0] = cv2.fillPoly(mask[:, :, 0].copy(), seg1, 255, lineType=cv2.LINE_AA)
                # color_mask = Image.fromarray(mask)
                # overlay = Image.blend(Image.fromarray(image), color_mask, 0.2)
                # display(overlay)
                # # disply generated images
                # mask = np.zeros((image.shape[0], image.shape[1], 3)).astype(np.uint8)
                # x, y, w, h = transformed_bboxes
                # p1, p2 = (int(x), int(y)), (int(x+w), int(y+h))
                # mask[:, :, 1] = cv2.rectangle(mask[:, :, 1].copy(), p1, p2, color=255, thickness=4)
                # seg1 = np.array([np.array(poly) for poly in segs_after_transform])# this has to be a list of arrays, each array is an annotation   
                # mask[:, :, 0] = cv2.fillPoly(mask[:, :, 0].copy(), seg1, 255, lineType=cv2.LINE_AA)
                # color_mask = Image.fromarray(mask)
                # overlay = Image.blend(Image.fromarray(transformed_image), color_mask, 0.2)
                # display(overlay)
    
            except:
                pass
    with open(out_put_dir+'training_defect_library_annotations.json', 'w') as f:
        json.dump(defect_annotations, f)  


# This is a data augmentation funtion, aims to augment current defects 
# the augmentation is applied on both the defect and its masks, therefore, it will preserve the defect mask
# It will generate an augmented defect library, as well as an annotation file, named
# augmented_training_defect_library_annotations.json
def augment(defects_path, defect_annotation_file, output_dir):
    # data augmentation for the target defects library
    # Nondestructive HorizontalFlip, VerticalFlip, Transpose, RandomRotate90
    # Non-rigid transformations: ElasticTransform, GridDistortion, OpticalDistortion
    # https://albumentations.ai/docs/examples/example_kaggle_salt/
    # defects_path: the path to prepared defects
    # defect_annotation_file: defect_annotation_file for prepared defects
    # output_dir: output_dir for gaugmented defects and its annotations

    # make output_dir
    os.makedirs(output_dir, exist_ok= True)
    # read defect_annotations 
    with open(defect_annotation_file, 'r') as f:
        defect_annotations = json.load(f)
    defects_list = os.listdir(defects_path)
    defects_list.remove('training_defect_library_annotations.json')

    augmented_training_defect_library_annotations={} # augmented_training_defect_library_annotations
    # try: # some annotation maynot have segmentation annotation, just skip
    count = 0
    for defect_name in tqdm(sorted(defects_list), desc = 'augmenting_defect_images', position = 0):
        for rept in range(10):
            #initilize initial defect annotation
            individual_defect_annotations = {}
            # get defect image as well as defect annotation
            defect_img = cv2.imread(os.path.join(defects_path, defect_name))
            defect_img = cv2.cvtColor(defect_img, cv2.COLOR_BGR2RGB)
            defct_annotation = defect_annotations[defect_name[:-4]][0]
            # get_defect_parameters
            height, width = defect_img.shape[0], defect_img.shape[1]
            x, y, w, h = defct_annotation["bbox"]
            centering_dic = {}
            centering_dic['x'] = -int(x+w/2-width/2)
            centering_dic['y'] = -int(y+h/2-height/2)
            bbox =  [defct_annotation["bbox"]]
            seg1 = [poly for poly in defct_annotation['seg1']]
            class_labels = [defct_annotation['category']]
            # here work with the mask instad of plgon since more transformrations can be applied
            # convert to mask
            defect_mask = plgon2msk(defct_annotation['seg1'],defect_img.shape)
            # visualize(defect_img, defect_mask)
            # define transformation
            # aug =  A.Compose([A.RandomContrast(p=1)#HorizontalFlip(p=1)#RandomRotate90(p=1)#VerticalFlip(p=1)#
            #             ], 
            #             bbox_params=A.BboxParams(format="coco", label_fields=["class_labels"]),
            #         )
            #augmentation
            aug = A.Compose([
                A.OneOf([
                    # A.RandomSizedCrop(min_max_height=(50, 101), height=height, width=width, p=0.5),
                    # A.PadIfNeeded(min_height=height, min_width=width, p=0.5)
                ], p=1),    
                A.VerticalFlip(p=0.5),              
                A.RandomRotate90(p=0.5),
                A.OneOf([
                    A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
                    A.GridDistortion(p=0.5),
                    # A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=1)                  
                    ], p=0.8),
                A.CLAHE(p=0.8),
                A.RandomBrightnessContrast(p=0.8),    
                A.RandomGamma(p=0.8)],
                bbox_params=A.BboxParams(format="coco", label_fields=["class_labels"]),)
            
            
            augmented = aug(image=defect_img, mask = defect_mask, bboxes=bbox, class_labels=class_labels)
            transformed_image = augmented["image"]
            transformed_bboxes = augmented["bboxes"][0]
            transformed_mask = augmented["mask"]
            # visualize(transformed_image, transformed_mask)
            segs_after_transform = msk2plgon(transformed_mask)
            # visualize(transformed_image, plgon2msk(segs_after_transform,transformed_image.shape))
            bbox = [ int(pt) for pt in transformed_bboxes]
            catgry = defct_annotation['category']

            individual_defect_annotations['bbox'] = bbox
            individual_defect_annotations['seg1'] = segs_after_transform
            individual_defect_annotations['category'] = catgry
            augmented_training_defect_library_annotations[str(count)] = [individual_defect_annotations]
            Image.fromarray(transformed_image).save(output_dir +str(count)+'.png')
            count = count + 1

            # # disply raw images
            # x, y, w, h = defct_annotation["bbox"]
            # p1, p2 = (int(x), int(y)), (int(x+w), int(y+h))
            # mask = np.zeros((defect_img.shape[0], defect_img.shape[1], 3)).astype(np.uint8)
            # mask[:, :, 1] = cv2.rectangle(mask[:, :, 1].copy(), p1, p2, color=255, thickness=4)
            # seg1 = [np.array(poly) for poly in defct_annotation['seg1']]
            # mask[:, :, 0] = cv2.fillPoly(mask[:, :, 0].copy(), seg1, 255, lineType=cv2.LINE_AA)
            # color_mask = Image.fromarray(mask)
            # overlay = Image.blend(Image.fromarray(defect_img), color_mask, 0.2)
            # display(overlay)
            
            # image = np.array(overlay)[y:y+h, x:x+w]
            # display(Image.fromarray(image))
            # os.makedirs('./Comp_res/'+'ps5_dataset/'+'demo/',exist_ok= True)
            # # Image.fromarray(image).save('./Comp_res/'+'ps5_dataset/'+'demo/'+str(count)+'_raw_w_msk.png')

            # mask[:, :, 0] =0
            # color_mask = Image.fromarray(mask)
            # overlay = Image.blend(Image.fromarray(defect_img), color_mask, 0.2)
            # image = np.array(overlay)[int(y):int(y+h), int(x):int(x+w)]
            # display(Image.fromarray(image))
            # Image.fromarray(image).save('./Comp_res/'+'ps5_dataset/'+'demo/'+str(count)+'_raw_w_o_msk.png')

            # disply generated images
            mask = np.zeros((transformed_image.shape[0], transformed_image.shape[1], 3)).astype(np.uint8)
            x, y, w, h = transformed_bboxes
            p1, p2 = (int(x), int(y)), (int(x+w), int(y+h))
            mask[:, :, 1] = cv2.rectangle(mask[:, :, 1].copy(), p1, p2, color=255, thickness=4)
            seg1 = np.array([np.array(poly) for poly in segs_after_transform])# this has to be a list of arrays, each array is an annotation   
            mask[:, :, 0] = cv2.fillPoly(mask[:, :, 0].copy(), seg1, 255, lineType=cv2.LINE_AA)
            color_mask = Image.fromarray(mask)
            overlay = Image.blend(Image.fromarray(transformed_image), color_mask, 0.2)
            # display(overlay)

            image = np.array(overlay)[int(y):int(y+h), int(x):int(x+w)]
            # display(Image.fromarray(image))
            Image.fromarray(image).save('./Comp_res/'+'ps5_dataset/'+'demo/'+str(count)+'_augmented_w_msk.png')
            
            mask[:, :, 0] =0
            color_mask = Image.fromarray(mask)
            overlay = Image.blend(Image.fromarray(transformed_image), color_mask, 0.2)
            image = np.array(overlay)[int(y):int(y+h), int(x):int(x+w)]
            # display(Image.fromarray(image))
            Image.fromarray(image).save('./Comp_res/'+'ps5_dataset/'+'demo/'+str(count)+'_augmented_w_o_msk.png')

        # display only the defects

        
            # save generated image
        # Image.fromarray(output.astype(np.uint8)).save('./Comp_res/'+'ps5_dataset/'+'training_set_v1/'+image_name)
        with open(output_dir+'augmented_training_defect_library_annotations.json', 'w') as f:
            json.dump(augmented_training_defect_library_annotations, f)

# This is a composition funtion, it will take input background iamges from "input_background_images_list" and 
# composite it with defects sampled from "defects_path"
# it will also generate the correpsonding defect annotation file, named 
# "generated_training_images_annotation.json"
# 
def generate_new_dataset(image_path, input_background_images_list, input_image_annotation, defects_path, defect_annotation_file, output_path, num_def_per_img = 5 ,agument_per_image = 10):
    # image_path: input iamge path
    # input_image_annotation: if empty just use [] (we do not reuqire the input image to be clean, if it has defect, please also include the 
    #                           defect annotations)
    # input_background_images_list: names of input_background_images
    # defects_path: path to augmented defect images
    # defect_annotation_file: 
    # output_path: 
    # num_def_per_img: number of generated defcts per image
    # agument_per_image: out_put_image_number/input_image_number, defalut 10
    # read defecta nnotation files
    os.makedirs(output_path, exist_ok= True)
    with open(defect_annotation_file, 'r') as f:
        defect_annotations = json.load(f)
    defects_list = os.listdir(defects_path)
    generated_training_images_annotation={} # annotations of the generated training set
    for image_name in tqdm(input_background_images_list, desc= 'Generating_synthetic_defect_image_dataset', position = 0):
        # generate defect on top of input images (repeat 10 times)
        for image_count in tqdm(range(agument_per_image), desc = 'synthetic images per input images', position = 1, leave=False):
            generated_image_name =  str(image_count)+'_'+image_name
            generated_training_images_annotation[generated_image_name] = input_image_annotation[image_name] # get the defct annotation already exist on the input background image
            # read tarining image
            image = cv2.imread(os.path.join(image_path, image_name))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width = image.shape[0], image.shape[1]
            # read defects from defect library
            # defects_selected = random.choice(sorted(images_names_train), num_def_per_img)
            count = 0 
            while count<num_def_per_img:
                try: # some annotation maynot have segmentation annotation, just skip
                    individual_defect_annotations = {}
                    # get defect image as well as defect annotation
                    defect_name = random.choice(sorted(defects_list))
                    defect_img = cv2.imread(os.path.join(defects_path, defect_name))
                    defect_img = cv2.cvtColor(defect_img, cv2.COLOR_BGR2RGB)
                    defct_annotation = defect_annotations[defect_name[:-4]][0]
                    x, y, w, h = defct_annotation["bbox"] # defect area (w, h)
                    seg1 = [np.array(poly) for poly in defct_annotation['seg1']]
                    if len(seg1)<1:
                        continue
                    # disply loaded defects
                    show_annotation(defect_img, [defct_annotation])

                    # define mask
                    mask = np.zeros((image.shape[0], image.shape[1])).astype(np.uint8)
                    mask[:, :] = cv2.fillPoly(mask[:, :].copy(), seg1, 255, lineType=cv2.LINE_AA) # using pixelwise accurate mask
                    # rectag = [np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]])]
                    # mask[:, :] = cv2.fillPoly(mask[:, :].copy(), rectag, 255, lineType=cv2.LINE_AA) # using box annotation mask
                    mask_img = np.asarray(mask).copy().astype(np.uint8)
                    # defnie soruce and traget image
                    source_img = np.asarray(defect_img).copy().astype(np.uint8)
                    source_img_height, source_img_width = source_img.shape[0], source_img.shape[1]
                    target_img = np.asarray(image).copy().astype(np.uint8) # the tagret image to paste defect
                    # define center defect location on the target image


                    # use self poission composition
                    # sample new location on the target image
                    new_x = random.choice(range(20, width-w-20))
                    new_y = random.choice(range(20, height-h-20))
                    loc = (new_x,new_y)
                    # move distance
                    Dx = new_x - x
                    Dy = new_y - y
                    if count == 0:
                        output = PoisImgComp(source_img, mask, loc, target_img,  defct_annotation["bbox"])
                    output = PoisImgComp(source_img, mask, loc, output,  defct_annotation["bbox"])
                    # use CV2 package
                    # move annotation according to the defect center location
                    # bbox
                    bbox_new = [new_x, new_y, w, h]
                    # seg 
                    seg1_new = [ (np.array(poly)+np.array([Dx,Dy])).tolist() for poly in defct_annotation['seg1']]
                    # add the annotation of this specific defect into annotation list
                    individual_defect_annotations['bbox'] = bbox_new
                    individual_defect_annotations['seg1'] = seg1_new
                    individual_defect_annotations['category'] = defct_annotation['category']
                    generated_training_images_annotation[generated_image_name].append(individual_defect_annotations)         
                    count = count + 1
                    # except:
                    #     pass
                    #     # use CV2 package
                    #     center = (int(width/2) + random.choice(range(-int(width/2)+int(w/2)+1,int(width/2)-int(w/2)-1)), int(height/2) + random.choice(range(-int(height/2)+int(h/2)+1,int(height/2)-int(h/2)-1)))
                    #     if count == 0:
                    #         output = cv2.seamlessClone(source_img, target_img.copy(), mask_img, center, cv2.NORMAL_CLONE)# cv2.MIXED_CLONE)
                    #     output = cv2.seamlessClone(source_img, output.copy(), mask_img, center, cv2.NORMAL_CLONE)
            

                    #     # move annotation according to the defect center location
                    #     # bbox
                    #     x_loc_source = x - int(source_img_width/2)
                    #     y_loc_source = y - int(source_img_height/2)
                    #     x_new = center[0] + x_loc_source
                    #     y_new = center[1] + y_loc_source
                    #     bbox_new = [x_new, y_new, w, h]
                    #     # seg 
                    #     seg1_new = [ (np.array(poly)-np.array([int(source_img_width/2),int(source_img_height/2)]) + np.array(center)).tolist() for poly in defct_annotation['seg1']]
                    #     # add the annotation of this specific defect into annotation list
                    #     individual_defect_annotations['bbox'] = bbox_new
                    #     individual_defect_annotations['seg1'] = seg1_new
                    #     individual_defect_annotations['category'] = defct_annotation['category']
                    #     generated_training_images_annotation[image_name].append(individual_defect_annotations)         
                    #     count = count + 1
                except:
                    pass
            # disply generated image with annotation
            save_path = output_path[:-1] + 'demo/'+ generated_image_name
            os.makedirs(output_path[:-1] + 'demo/',exist_ok = True)
            show_annotation(output,generated_training_images_annotation[generated_image_name], save_path)
            # disply generated image
            # display(Image.fromarray(output))      
            #save generated image
            Image.fromarray(output.astype(np.uint8)).save(output_path+generated_image_name)
    with open(output_path+'generated_training_images_annotation.json', 'w') as f:
        json.dump(generated_training_images_annotation, f)

# Those are helper functions
# convert from polygone to mask
def plgon2msk(polygons, mask_shape):
    # input plgons in the format of 
    # [
    # [[x1,y1], [x2,y2],..], contour one
    #  [[x1,y1], [x2,y2],..], contour two
    # ]
    # input mask shape (x,y,c) last one is channels
    # output binary mask
    mask = np.zeros((mask_shape[0], mask_shape[1])).astype(np.uint8) 
    seg1 = [np.array(poly) for poly in polygons]
    mask = cv2.fillPoly(mask.copy(), seg1, 255, lineType=cv2.LINE_AA)
    return mask
# covert from mask to polygone
def msk2plgon(mask):
    # input binary mask
    # output plgons in the format of 
    # [
    # [[x1,y1], [x2,y2],..], contour one
    #  [[x1,y1], [x2,y2],..], contour two
    # ]
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    polygons = []
    for object in contours:
        coords = []
        for point in object:
            coords.append([int(point[0][0]),int(point[0][1]) ])
        polygons.append(coords)
        # polygons = [np.ar(pts, np.int32).reshape(-1, 2) for pts in polygons]
    return polygons
    
def visualize(image, mask, original_image=None, original_mask=None):
    fontsize = 18
    
    if original_image is None and original_mask is None:
        f, ax = plt.subplots(2, 1, figsize=(8, 8))

        ax[0].imshow(image)
        ax[1].imshow(mask)
    else:
        f, ax = plt.subplots(2, 2, figsize=(8, 8))

        ax[0, 0].imshow(original_image)
        ax[0, 0].set_title('Original image', fontsize=fontsize)
        
        ax[1, 0].imshow(original_mask)
        ax[1, 0].set_title('Original mask', fontsize=fontsize)
        
        ax[0, 1].imshow(image)
        ax[0, 1].set_title('Transformed image', fontsize=fontsize)
        
        ax[1, 1].imshow(mask)
        ax[1, 1].set_title('Transformed mask', fontsize=fontsize)
        
def show_annotation(image, annotations, save_path=[]):
    # image: with defect
    # annotation: defect annotations on this image (should be multiple, if only one should wrap with [])
    # save_path: path to save the new image with annotation
    mask = np.zeros((image.shape[0], image.shape[1], 3)).astype(np.uint8)
    for individual_defect_annotations in annotations:
        x, y, w, h = individual_defect_annotations['bbox']
        p1, p2 = (int(x), int(y)), (int(x+w), int(y+h))
        mask[:, :, 1] = cv2.rectangle(mask[:, :, 1].copy(), p1, p2, color=255, thickness=4)
        seg1 = np.array([np.array(poly) for poly in individual_defect_annotations['seg1']])# this has to be a list of arrays, each array is an annotation   
        mask[:, :, 0] = cv2.fillPoly(mask[:, :, 0].copy(), seg1, 255, lineType=cv2.LINE_AA)
    color_mask = Image.fromarray(mask)
    overlay = Image.blend(Image.fromarray(image), color_mask, 0.2)
    # display(overlay)
    if save_path:
        overlay.save(save_path)
    return