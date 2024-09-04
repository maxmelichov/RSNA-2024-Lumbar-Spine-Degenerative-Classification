import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pandas as pd
import numpy as np
import albumentations as A
import pydicom
import os
from PIL import Image
import sys
sys.path.insert(0, r'F:\Projects\Kaggle\RSNA-2024-Lumbar-Spine-Degenerative-Classification\preprocessing')
from segmantation_inference import SegmentaionInference
from Cross_Reference import CrossReference
from detection_inference import DetectionInference, transforms
import matplotlib.pyplot as plt
import re
import warnings
warnings.filterwarnings("ignore")

COUNT = 0 
class CFG():
    AUG_PROB = 0.75
    NOT_DEBUG = True
    AUG = True
    Axial_shape = (152, 152)
    Sagittal_shape = (152, 152)
    channel_size_sagittal = 20
    channel_size_axial = 9
    train_path = "train_images"
    segmentation = SegmentaionInference(model_path=r"weights\simple_unet.onnx")
    DetectionInference = DetectionInference(model_path=r"weights\axial_detection_resnet18.pth", transforms=transforms)
    label2id = {'Normal/Mild': 0, 'Moderate':1, 'Severe':2, np.nan: -100}
    category2id = {"L1": 1, "L2": 2, "L3": 3, "L4": 4, "L5": 5, "L5-S1": 11, "L4-L5": 12, "L3-L4": 13, "L2-L3": 14, "L1-L2": 15}
    skip_study_id = [2492114990, 2780132468, 3008676218]
    two_classes_category = {11: 'L5-S1', 12: 'L4-L5', 13: 'L3-L4', 14: 'L2-L3', 15: 'L1-L2'}

cfg = CFG()
cross_reference = CrossReference()

transforms_axial = A.Compose([
    A.Resize(cfg.Axial_shape[0], cfg.Axial_shape[1]),
    A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=cfg.AUG_PROB),
    A.Perspective(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.OneOf([
        A.MotionBlur(blur_limit=5),
        A.MedianBlur(blur_limit=5),
        A.GaussianBlur(blur_limit=5),
        A.GaussNoise(var_limit=(5.0, 30.0)),
    ], p=cfg.AUG_PROB),

    # A.OneOf([
    #     A.OpticalDistortion(distort_limit=1.0),
    #     A.GridDistortion(num_steps=5, distort_limit=1.),
    #     A.ElasticTransform(alpha=3),
    # ], p=0.5),

    # A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=cfg.AUG_PROB),
    A.CoarseDropout(max_holes=16, max_height=16, max_width=16, min_holes=1, min_height=4, min_width=4, p=cfg.AUG_PROB),
    A.Normalize(mean=0.5, std=0.5)
])

transforms_sagittal = A.Compose([
    A.Resize(cfg.Sagittal_shape[0], cfg.Sagittal_shape[1]),
    A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=cfg.AUG_PROB),
    A.Perspective(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.OneOf([
        A.MotionBlur(blur_limit=5),
        A.MedianBlur(blur_limit=5),
        A.GaussianBlur(blur_limit=5),
        A.GaussNoise(var_limit=(5.0, 30.0)),
    ], p=cfg.AUG_PROB),

    # A.OneOf([
    #     A.OpticalDistortion(distort_limit=1.0),
    #     A.GridDistortion(num_steps=5, distort_limit=1.),
    #     A.ElasticTransform(alpha=3),
    # ], p=0.5),

    # A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=cfg.AUG_PROB),
    A.CoarseDropout(max_holes=16, max_height=16, max_width=16, min_holes=1, min_height=4, min_width=4, p=cfg.AUG_PROB),
    A.Normalize(mean=0.5, std=0.5)
])

transforms_val = A.Compose([
    A.Normalize(mean=0.5, std=0.5)
])

class CustomDataset(Dataset):
    def __init__(self, train_data, labels_path, description_path, transforms_sagittal=None, transforms_axial=None):
        self.df = pd.read_csv(train_data)
        self.df_labels = pd.read_csv(labels_path)
        self.df_description = pd.read_csv(description_path)
        self.transforms_sagittal = transforms_sagittal
        self.transforms_axial = transforms_axial
        self.train_path = "train_images"
        pass
    def __len__(self):
        return len(self.df_labels)


    @staticmethod
    def plot(stack,x = 5,y = 6):
        fig, axes = plt.subplots(x, y, figsize=(15, 9))
        for i, ax in enumerate(axes.flat):
            ax.imshow(stack[..., i], cmap='gray')
        plt.tight_layout()
        plt.show()

    def load_dicom(self, path):
        original_dicom = pydicom.dcmread(path).pixel_array
        original_dicom = np.array(self.resize_image(original_dicom, (512, 512)))
        original_dicom = (original_dicom - original_dicom.min()) / (original_dicom.max() - original_dicom.min() + 1e-6) * 255
        original_dicom = original_dicom.astype(np.uint8)
        lower, upper = np.percentile(original_dicom, (1, 99))
        original_dicom = np.clip(original_dicom, lower, upper)
        return original_dicom

    @staticmethod
    def pad_images_list(images_list, max_len): # need to check this function
        if len(images_list) <= 0:
            raise ValueError("images_list is empty")
        if len(images_list) == max_len:
            return images_list
        
        n = len(images_list)
        output_list = []
        
        # How many times should we duplicate each element minimally?
        min_repeats = max_len // n
        
        # How many extra duplicates are needed beyond minimal repeats?
        extra = max_len % n
        
        # Determine the central region to duplicate more
        mid_point = n // 2
        start_extra = mid_point - (extra // 2)
        end_extra = start_extra + extra
        
        # Duplicate elements, adding extra repeats to central elements
        for i in range(n):
            repeats = min_repeats + 1 if start_extra <= i < end_extra else min_repeats
            output_list.extend([images_list[i]] * repeats)

        return output_list
    
    @staticmethod
    def unpad_images_list(images_list, max_len): # need to check this function
        i = 0
        while len(images_list) > max_len:
            if i % 2 == 0:
                images_list.pop(-1)
            else:
                images_list.pop(0)
            i += 1
        return images_list
        
    def center_crop_by_category(self, pixel_array, bboxes, category):
        global COUNT
        bbox = bboxes[category]
        x, y, h, w = bbox[0]
        if x == -1 or y == -1 or h == -1 or w == -1:
            COUNT+=1
            print(COUNT)
        image = Image.fromarray(pixel_array)
        margin = 152 // 2 
        cropped_image = image.crop((x - 20, y - margin, x + 152 - 20, y + margin))
        return np.array(cropped_image)

    

    @staticmethod
    def resize_image(pixel_array, new_size):
        if pixel_array.shape == new_size:
            return pixel_array
        else:
            image = Image.fromarray(pixel_array)
            return image.resize((new_size[1], new_size[0]))

    @staticmethod
    def extract_number(filename):
            match = re.search(r'\d+', filename)
            return int(match.group()) if match else 0
    
    @staticmethod
    def crop_axial_center(image, bbox):
                min_x, max_x = 999999, -1
                min_y, max_y = 999999, -1
                for x, y, h, w in bbox:
                    min_x = min(min_x, x)
                    max_x = max(max_x, x+w)
                    min_y = min(min_y, y)
                    max_y = max(max_y, y+h)
                if type(image) != Image.Image:
                    image = Image.fromarray(image)
                
                if min_x == 999999 or min_y == 999999:
                    
                    width, height = image.size
                    # Define the size of the crop
                    crop_size = 304

                    # Calculate coordinates for the middle crop
                    left = (width - crop_size) // 2
                    top = (height - crop_size) // 2
                    right = left + crop_size
                    bottom = top + crop_size
                    cropped_image = image.crop((left, top, right, bottom))
                    # print("left: ", left, "top: ", top, "right: ", right, "bottom: ", bottom)
                    return (
                    cropped_image.crop((0, 0, 152, crop_size)),   # Adjusted coordinates
                    cropped_image.crop((76, 0, 228, crop_size)),  # Adjusted coordinates
                    cropped_image.crop((152, 0, 304, crop_size))  # Adjusted coordinates
                )
                
                # Crop the center of the image
                margin = 304 // 2 
                
                return (image.crop((min_x - margin, min_y - 50, min_x, min_y + 102)),
                        image.crop((min_x - 76 , min_y - 50, min_x + 76, min_y + 102)),
                        image.crop((min_x, min_y - 50, min_x + margin, min_y + 102)))

    def crop_sagittal_center(self, file, Sagittal_bboxes_scaled, Sagittal_path, two_classes_category):

            original_dicom = self.load_dicom(os.path.join(Sagittal_path, file))

            new_pixel_array = self.center_crop_by_category(original_dicom, Sagittal_bboxes_scaled, cfg.category2id[two_classes_category])
            # resized_pixel_array = self.resize_image(new_pixel_array, (cfg.Sagittal_shape[0], cfg.Sagittal_shape[1]))
            new_shape = new_pixel_array.shape
            # Ensure the padded array is large enough to hold new_pixel_array
            padded_array = np.zeros((max(cfg.Sagittal_shape[0], new_shape[0]), max(cfg.Sagittal_shape[1], new_shape[1])))
            
            # Compute the starting indices for centering the new_pixel_array
            start_x = (padded_array.shape[0] - new_shape[0]) // 2
            start_y = (padded_array.shape[1] - new_shape[1]) // 2

            # Place the new_pixel_array in the center of the padded_array
            padded_array[start_x:start_x + new_shape[0], start_y:start_y + new_shape[1]] = new_pixel_array
            return padded_array[:cfg.Sagittal_shape[0], :cfg.Sagittal_shape[1]].astype(np.uint8)



        


    def create_stack(self, sagittal_stack, axial_stack,
                     Sagittal_T2_files, Sagittal_T2_bboxes, Sagittal_T2_path,
                        Sagittal_T1_files, Sagittal_T1_bboxes, Sagittal_T1_path,
                     category, two_classes_category, df_classes):
        k = 0
        
        if len(Sagittal_T1_files) < cfg.channel_size_sagittal // 2:
            Sagittal_T1_files = self.pad_images_list(Sagittal_T1_files, cfg.channel_size_sagittal//2)
        elif len(Sagittal_T1_files) > cfg.channel_size_sagittal // 2:
            Sagittal_T1_files = self.unpad_images_list(Sagittal_T1_files, cfg.channel_size_sagittal//2)

        original_dicom = pydicom.dcmread(os.path.join(Sagittal_T1_path, Sagittal_T1_files[len(Sagittal_T1_files)//2])).pixel_array
        # scale the bboxes to the new size of the image after resizing it to 512x512
        Sagittal_bboxes_scaled = cfg.segmentation.scale_bboxes(Sagittal_T1_bboxes, (512, 512), original_dicom.shape)
        for file in Sagittal_T1_files:
            sagittal_stack[..., k] = self.crop_sagittal_center(file, Sagittal_bboxes_scaled, Sagittal_T1_path, two_classes_category)
            k += 1
        


        if len(Sagittal_T2_files) < cfg.channel_size_sagittal // 2:
            Sagittal_T2_files = self.pad_images_list(Sagittal_T2_files, cfg.channel_size_sagittal//2)
        elif len(Sagittal_T2_files) > cfg.channel_size_sagittal // 2:
            Sagittal_T2_files = self.unpad_images_list(Sagittal_T2_files, cfg.channel_size_sagittal//2)
        original_dicom = pydicom.dcmread(os.path.join(Sagittal_T2_path, Sagittal_T2_files[len(Sagittal_T2_files)//2])).pixel_array
        Sagittal_T2_bboxes = cfg.segmentation.scale_bboxes(Sagittal_T2_bboxes, (512, 512), original_dicom.shape)
        for file in Sagittal_T2_files:
            sagittal_stack[..., k] = self.crop_sagittal_center(file, Sagittal_T2_bboxes, Sagittal_T2_path, two_classes_category)
            k += 1
        
        

        l = df_classes['path'].loc[(df_classes['class_id'] == category) | (df_classes['class_id'] == two_classes_category)].unique()

        l = l.tolist()
        if len(l) == 0:
            l = df_classes['path'].loc[(df_classes['class_id'] == category)].unique()
            l = l.tolist()
        l = sorted(l, key=self.extract_number)

        if len(l) == 0:
            return sagittal_stack, axial_stack
        
        if len(l) < 3:
            l = self.pad_images_list(l, 3)

        elif len(l) > 3:
            l = self.unpad_images_list(l, 3)
        
        
        p = 0
        j = 3
        o = 6
        for file in l:
            original_dicom = pydicom.dcmread(file).pixel_array
            original_dicom = original_dicom.clip(np.percentile(original_dicom, 1), np.percentile(original_dicom, 99))
            bbox = cfg.DetectionInference.inference(original_dicom, 512, 512)
            original_dicom = (original_dicom - original_dicom.min()) / (original_dicom.max() - original_dicom.min() + 1e-6) * 255
            original_dicom = original_dicom.astype(np.uint8)
            resized_pixel_array = self.resize_image(original_dicom, (512,512))
            # Crop the center of the DICOM image
            cropped_left, cropped_middle, cropped_right = self.crop_axial_center(resized_pixel_array, bbox)
            cropped_left = self.resize_image(np.array(cropped_left), (cfg.Axial_shape[0], cfg.Axial_shape[1]))
            cropped_middle = self.resize_image(np.array(cropped_middle), (cfg.Axial_shape[0], cfg.Axial_shape[1]))
            cropped_right = self.resize_image(np.array(cropped_right), (cfg.Axial_shape[0], cfg.Axial_shape[1]))
            axial_stack[..., p] = np.array(cropped_left).astype(np.uint8)
            sagittal_stack[..., k + p] = np.array(cropped_left).astype(np.uint8)
            p += 1
            axial_stack[..., j] = np.array(cropped_middle).astype(np.uint8)
            sagittal_stack[..., k + j] = np.array(cropped_middle).astype(np.uint8)
            j += 1
            axial_stack[..., o] = np.array(cropped_right).astype(np.uint8)
            sagittal_stack[..., k + o] = np.array(cropped_right).astype(np.uint8)
            o += 1
        
        return sagittal_stack, axial_stack

    
    @staticmethod
    def _is_all_black(image_array):
        return np.all(image_array == 0)

    @staticmethod
    def _reorder_of_labels(labels):
        # Create a list to hold the reordered labels
        reordered_labels = []

        # Loop over the remainders from 0 to 4
        for i in range(5):
            # Add labels to reordered_labels based on the remainder when index is divided by 5
            reordered_labels.extend([label for idx, label in enumerate(labels) if idx % 5 == i])
        return torch.tensor(reordered_labels)
    
    def __getitem__(self, index):
        sagittal_l1_l2 = np.zeros((cfg.Sagittal_shape[0], cfg.Sagittal_shape[1], cfg.channel_size_sagittal + cfg.channel_size_axial), dtype = np.uint8)
        axial_l1_l2 = np.zeros((cfg.Axial_shape[0], cfg.Axial_shape[1], cfg.channel_size_axial), dtype = np.uint8)
        sagittal_l2_l3 = np.zeros((cfg.Sagittal_shape[0], cfg.Sagittal_shape[1], cfg.channel_size_sagittal + cfg.channel_size_axial), dtype = np.uint8)
        axial_l2_l3 = np.zeros((cfg.Axial_shape[0], cfg.Axial_shape[1], cfg.channel_size_axial), dtype = np.uint8)
        sagittal_l3_l4 = np.zeros((cfg.Sagittal_shape[0], cfg.Sagittal_shape[1], cfg.channel_size_sagittal + cfg.channel_size_axial), dtype = np.uint8)
        axial_l3_l4 = np.zeros((cfg.Axial_shape[0], cfg.Axial_shape[1], cfg.channel_size_axial), dtype = np.uint8)
        sagittal_l4_l5 = np.zeros((cfg.Sagittal_shape[0], cfg.Sagittal_shape[1], cfg.channel_size_sagittal + cfg.channel_size_axial), dtype = np.uint8)
        axial_l4_l5 = np.zeros((cfg.Axial_shape[0], cfg.Axial_shape[1], cfg.channel_size_axial), dtype = np.uint8)
        sagittal_l5_s1 = np.zeros((cfg.Sagittal_shape[0], cfg.Sagittal_shape[1], cfg.channel_size_sagittal + cfg.channel_size_axial), dtype = np.uint8)
        axial_l5_s1 = np.zeros((cfg.Axial_shape[0], cfg.Axial_shape[1], cfg.channel_size_axial), dtype = np.uint8)

        
        study_id = self.df_labels.iloc[index]['study_id']

        if study_id in cfg.skip_study_id:
            return self.__getitem__((index + 1) % len(self.df_labels))  # Try next item, wrap around if at end
    
        sub_set = self.df_labels[(self.df_labels['study_id'] == study_id)]

        @staticmethod
        def _count_neg_ones(bboxes):
            count = 0
            for vals in bboxes.values():
                count += vals.count((-1, -1, -1, -1))
            return count
        
        for col in sub_set.columns:
            if col == "general_path_to_Sagittal_T1":
                Sagittal_T1_path = sub_set[col].iloc[0]
                Sagittal_T1_files = os.listdir(Sagittal_T1_path)
                Sagittal_T1_files = sorted(Sagittal_T1_files, key=self.extract_number)
                middle_index = len(Sagittal_T1_files) // 2
                Sagittal_T1_bboxes = cfg.segmentation.inference(os.path.join(Sagittal_T1_path, Sagittal_T1_files[middle_index]))
                if _count_neg_ones(Sagittal_T1_bboxes) != 0:
                    pmin = len(Sagittal_T1_files) // 2 - 5
                    pmax = len(Sagittal_T1_files) // 2 + 5
                    for p in range(pmin, pmax):
                        temp = cfg.segmentation.inference(os.path.join(Sagittal_T1_path, Sagittal_T1_files[p]))
                        for key, value in temp.items():
                            if value != [(-1, -1, -1, -1)]:
                                if key not in Sagittal_T1_bboxes or Sagittal_T1_bboxes[key] == [(-1, -1, -1, -1)]:
                                    Sagittal_T1_bboxes[key] = value
                        if _count_neg_ones(Sagittal_T1_bboxes) == 0:
                            break

            
            elif col == "general_path_to_Sagittal_T2":
                Sagittal_T2_path = sub_set[col].iloc[0]
                Sagittal_T2_files = os.listdir(Sagittal_T2_path)
                Sagittal_T2_files = sorted(Sagittal_T2_files, key=self.extract_number)
                middle_index = len(Sagittal_T2_files) // 2
                Sagittal_T2_bboxes = cfg.segmentation.inference(os.path.join(Sagittal_T2_path, Sagittal_T2_files[middle_index]))
                if _count_neg_ones(Sagittal_T2_bboxes) != 0:
                    pmin = len(Sagittal_T2_files) // 2 - 5
                    pmax = len(Sagittal_T2_files) // 2 + 5
                    for p in range(pmin, pmax):
                        temp = cfg.segmentation.inference(os.path.join(Sagittal_T2_path, Sagittal_T2_files[p]))
                        for key, value in temp.items():
                            if value != [(-1, -1, -1, -1)]:
                                if key not in Sagittal_T2_bboxes or Sagittal_T2_bboxes[key] == [(-1, -1, -1, -1)]:
                                    Sagittal_T2_bboxes[key] = value
                        if _count_neg_ones(Sagittal_T2_bboxes) == 0:
                            break
        
        decription_df = self.df_description[(self.df_description['study_id'] == study_id)]
        # df_classes = self.divide_Axiel(decription_df)
        
        df_classes = cross_reference.get_cross_reference_for_Axial(decription_df, "test")
        # print(df_classes)

            
        
        sagittal_l1_l2, axial_l1_l2 = self.create_stack(sagittal_l1_l2, axial_l1_l2,
                                                        Sagittal_T2_files, Sagittal_T2_bboxes, Sagittal_T2_path,
                                                        Sagittal_T1_files, Sagittal_T1_bboxes, Sagittal_T1_path,
                                                        "L1", "L1-L2", df_classes)
        
        sagittal_l2_l3, axial_l2_l3 = self.create_stack(sagittal_l2_l3, axial_l2_l3,
                                                        Sagittal_T2_files, Sagittal_T2_bboxes, Sagittal_T2_path,
                                                        Sagittal_T1_files, Sagittal_T1_bboxes, Sagittal_T1_path,
                                        "L2", "L2-L3", df_classes)
        
        sagittal_l3_l4, axial_l3_l4 = self.create_stack(sagittal_l3_l4, axial_l3_l4,
                                                        Sagittal_T2_files, Sagittal_T2_bboxes, Sagittal_T2_path,
                                                        Sagittal_T1_files, Sagittal_T1_bboxes, Sagittal_T1_path,
                                        "L3", "L3-L4", df_classes)
                                        
        sagittal_l4_l5, axial_l4_l5 = self.create_stack(sagittal_l4_l5, axial_l4_l5,
                                                        Sagittal_T2_files, Sagittal_T2_bboxes, Sagittal_T2_path,
                                                        Sagittal_T1_files, Sagittal_T1_bboxes, Sagittal_T1_path,
                                        "L4", "L4-L5", df_classes)
        
        sagittal_l5_s1, axial_l5_s1 = self.create_stack(sagittal_l5_s1, axial_l5_s1,
                                                        Sagittal_T2_files, Sagittal_T2_bboxes, Sagittal_T2_path,
                                                        Sagittal_T1_files, Sagittal_T1_bboxes, Sagittal_T1_path,
                                        "L5", "L5-S1", df_classes)
        
        # self.plot(sagittal_l1_l2[...,], x = 4, y = 7)
        # self.plot(sagittal_l2_l3[...,], x = 4, y = 7)
        # self.plot(sagittal_l3_l4[...,], x = 4, y = 7)
        # self.plot(sagittal_l4_l5[...,], x = 4, y = 7)
        # self.plot(sagittal_l5_s1[...,], x = 4, y = 7)
        # self.plot(axial_l1_l2[...,],x= 1, y = 3)
        # self.plot(axial_l2_l3[...,],x= 1, y = 3)
        # self.plot(axial_l3_l4[...,],x= 1, y = 3)
        # self.plot(axial_l4_l5[...,],x= 1, y = 3)
        # self.plot(axial_l5_s1[...,],x= 1, y = 3)



        if self.transforms_sagittal and self.transforms_axial:
            sagittal_l1_l2 = self.transforms_sagittal(image=sagittal_l1_l2)['image']
            sagittal_l2_l3 = self.transforms_sagittal(image=sagittal_l2_l3)['image']
            sagittal_l3_l4 = self.transforms_sagittal(image=sagittal_l3_l4)['image']
            sagittal_l4_l5 = self.transforms_sagittal(image=sagittal_l4_l5)['image']
            sagittal_l5_s1 = self.transforms_sagittal(image=sagittal_l5_s1)['image']
            axial_l1_l2 = self.transforms_axial(image=axial_l1_l2)['image']
            axial_l2_l3 = self.transforms_axial(image=axial_l2_l3)['image']
            axial_l3_l4 = self.transforms_axial(image=axial_l3_l4)['image']
            axial_l4_l5 = self.transforms_axial(image=axial_l4_l5)['image']
            axial_l5_s1 = self.transforms_axial(image=axial_l5_s1)['image']

        # self.plot(sagittal_l1_l2[...,], x = 3, y = 4)
        # self.plot(sagittal_l2_l3[...,], x = 3, y = 4)
        # self.plot(sagittal_l3_l4[...,], x = 3, y = 4)
        # self.plot(sagittal_l4_l5[...,], x = 3, y = 4)
        # self.plot(sagittal_l5_s1[...,], x = 3, y = 4)
        # self.plot(axial_l1_l2[...,])
        # self.plot(axial_l1_l2[...,])
        # self.plot(axial_l1_l2[...,])
        # self.plot(axial_l1_l2[...,])
        # self.plot(axial_l1_l2[...,])
        sagittal_l1_l2 = torch.tensor(sagittal_l1_l2).permute(2, 0, 1).float()
        axial_l1_l2 = torch.tensor(axial_l1_l2).permute(2, 0, 1).float()
        sagittal_l2_l3 = torch.tensor(sagittal_l2_l3).permute(2, 0, 1).float()
        axial_l2_l3 = torch.tensor(axial_l2_l3).permute(2, 0, 1).float()
        sagittal_l3_l4 = torch.tensor(sagittal_l3_l4).permute(2, 0, 1).float()
        axial_l3_l4 = torch.tensor(axial_l3_l4).permute(2, 0, 1).float()
        sagittal_l4_l5 = torch.tensor(sagittal_l4_l5).permute(2, 0, 1).float()
        axial_l4_l5 = torch.tensor(axial_l4_l5).permute(2, 0, 1).float()
        sagittal_l5_s1 = torch.tensor(sagittal_l5_s1).permute(2, 0, 1).float()
        axial_l5_s1 = torch.tensor(axial_l5_s1).permute(2, 0, 1).float()


        t = self.df.iloc[index][1:]
        t = t.map(lambda y: cfg.label2id[y] if not pd.isna(y) else cfg.label2id[np.nan])
        labels = t.values.astype(np.int64)
        reordered_labels = self._reorder_of_labels(labels)
        # reordered_labels = torch.tensor([reordered_labels.tolist()] * 10)
        return (sagittal_l1_l2, axial_l1_l2, sagittal_l2_l3,
                 axial_l2_l3, sagittal_l3_l4, axial_l3_l4, sagittal_l4_l5, axial_l4_l5, sagittal_l5_s1,
                   axial_l5_s1, reordered_labels)

def data_loader(train_data: Path, labels_path: Path, description_path: Path, mode: str = "train") -> tuple[DataLoader, DataLoader]:
    """
    Loads the data for training.
    Args:
        train_data (Path): The path to the training data.
        labels_path (Path): The path to the labels.
        description_path (Path): The path to the description.
    Returns:
        tuple[DataLoader, DataLoader]: A tuple containing two DataLoaders.
    """
    if mode == "train":
        custom_train = CustomDataset(train_data, labels_path, description_path, transforms_sagittal=transforms_sagittal, transforms_axial=transforms_axial)
        return custom_train
    else:
        custom_val = CustomDataset(train_data, labels_path, description_path, transforms_sagittal=transforms_val, transforms_axial=transforms_val)
        return custom_val