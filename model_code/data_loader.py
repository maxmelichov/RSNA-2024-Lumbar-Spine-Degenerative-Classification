import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pandas as pd
import numpy as np
import albumentations as A
import pydicom
import os
from PIL import Image
from preprocessing.segmantation_inference import SegmentaionInference
import matplotlib.pyplot as plt
import re


class CFG():
    AUG_PROB = 0.75
    NOT_DEBUG = True
    AUG = True
    Axial_shape = (384, 384)
    Sagittal_shape = (192, 192)
    train_path = "train_images"
    segmentation = SegmentaionInference(model_path=r"weights\simple_unet.onnx")
    label2id = {'Normal/Mild': 0, 'Moderate':1, 'Severe':2, np.nan: -100}
    category2id = {"L1": 1, "L2": 2, "L3": 3, "L4": 4, "L5": 5, "L5-S1": 11, "L4-L5": 12, "L3-L4": 13, "L2-L3": 14, "L1-L2": 15}
    skip_study_id = [2492114990, 2780132468, 3008676218]
    two_classes_category = {11: 'L5-S1', 12: 'L4-L5', 13: 'L3-L4', 14: 'L2-L3', 15: 'L1-L2'}

cfg = CFG()


transforms_axial = A.Compose([
    A.Resize(384, 384),
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

    A.OneOf([
        A.OpticalDistortion(distort_limit=1.0),
        A.GridDistortion(num_steps=5, distort_limit=1.),
        A.ElasticTransform(alpha=3),
    ], p=0.5),

    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=cfg.AUG_PROB),
    A.CoarseDropout(max_holes=32, max_height=32, max_width=32, min_holes=1, min_height=8, min_width=8, p=cfg.AUG_PROB),  
    A.Normalize(mean=0.5, std=0.5)
])

transforms_sagittal = A.Compose([
    A.Resize(192, 192),
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

    A.CoarseDropout(max_holes=1, max_height=int(192*0.275), max_width=int(192*0.275), p=cfg.AUG_PROB),    
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
    def plot(stack):
        fig, axes = plt.subplots(5, 5, figsize=(15, 9))
        for i, ax in enumerate(axes.flat):
            ax.imshow(stack[..., i], cmap='gray')
        plt.tight_layout()
        plt.show()

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
    def center_crop(pixel_array, bboxes): # need to check this function
        min_x, max_x = 999999, -1
        min_y, max_y = 999999, -1
        list_of_bboxes = [boxes[0] for boxes in bboxes.values()]
        for x, y, h, w in list_of_bboxes:
            if x == -1 and y == -1 and h == -1 and w == -1:
                continue
            min_x = min(min_x, x)
            max_x = max(max_x, x+w)
            min_y = min(min_y, y)
            max_y = max(max_y, y+h)
        # Convert DICOM pixel array to PIL image
        image = Image.fromarray(pixel_array)
        # Crop the image using the calculated bounding box
        if min_x == 999999:
            min_x = 0
        if min_y == 999999:
            min_y = 0
        if max_x == -1:
            max_x = pixel_array.shape[1]
        if max_y == -1:
            max_y = pixel_array.shape[1]
        cropped_image_array = np.array(image.crop((min_x, min_y, max_x + 1, max_y + 1)))
        return cropped_image_array
    

    def center_crop_by_category(self, pixel_array, bboxes, category, second_category): # need to check this function
        bbox1 = bboxes[category]
        bbox2 = bboxes[second_category]
        x, y, h, w = bbox1[0]
        x2, y2, h2, w2 = bbox2[0]
        # Function to get the minimum value ignoring -1
        def min_ignore_neg_one(a, b):
            if a == -1:
                return b
            if b == -1:
                return a
            return min(a, b)

        # Function to get the maximum value ignoring -1
        min_x = min_ignore_neg_one(x, x2)
        min_y = min_ignore_neg_one(y, y2)
        max_x = max(x + w, x2 + w2)
        max_y = max(y + h, y2 + h2)
        image = Image.fromarray(pixel_array)
        if (min_x == -1 and min_y == -1) or ((max_x - min_x < 35) or (max_y - min_y < 35)):
            # shape = np.array(original_dicom).shape
            # need to add plt.imshow() of the segmentation mask
            return self.center_crop(pixel_array, bboxes)
        cropped_image = image.crop((min(0, min_x-50), min_y-10, max_x + 30, max_y))

        return np.array(cropped_image)
        
        

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

    def divide_Axiel(self, sub_set):
        df_classes = pd.DataFrame(columns=['path', 'class_id'])
        study_id = sub_set['study_id'].iloc[0]
        series_id_axial = sub_set['series_id'].loc[sub_set['series_description'] == "Axial T2"].iloc[0]
        list_ = os.listdir(os.path.join(self.train_path, str(study_id), str(series_id_axial)))
        list_ = sorted(list_, key=self.extract_number)
        divide_by_5 = len(list_) // 5
        remainder = len(list_) % 5

        class_ids = ["L1", "L2", "L3", "L4", "L5"]
        start_idx = 0

        for i, class_id in enumerate(class_ids):
            end_idx = start_idx + divide_by_5 + (1 if i < remainder else 0)  # Add 1 to the first 'remainder' groups
            for file in list_[start_idx:end_idx]:
                df_classes.loc[len(df_classes)] = [os.path.join(self.train_path, str(study_id), str(series_id_axial), file), class_id]
            start_idx = end_idx
        
        return df_classes


    def create_stack(self, sagittal_stack, axial_stack, Sagittal_T1_files, Sagittal_T1_path,
                     Sagittal_T2_files, Sagittal_T2_bboxes, Sagittal_T2_path,
                     category, two_classes_category, sub_set):

        k = 0
        
        
        if len(Sagittal_T2_files) < 15:
            Sagittal_T2_files = self.pad_images_list(Sagittal_T2_files, 15)
        elif len(Sagittal_T2_files) > 15:
            Sagittal_T2_files = self.unpad_images_list(Sagittal_T2_files, 15)
        for file in Sagittal_T2_files:
            original_dicom = pydicom.dcmread(os.path.join(Sagittal_T2_path, file)).pixel_array
            original_dicom = (original_dicom - original_dicom.min()) / (original_dicom.max() - original_dicom.min() + 1e-6) * 255
            new_pixel_array = self.center_crop_by_category(original_dicom, Sagittal_T2_bboxes,
                                                           cfg.category2id[category], cfg.category2id[two_classes_category])
            resized_pixel_array = self.resize_image(new_pixel_array, (192, 192))
            sagittal_stack[..., k] = resized_pixel_array
            k += 1

        k = 0
        if len(Sagittal_T1_files) < 15:
            Sagittal_T1_files = self.pad_images_list(Sagittal_T1_files, 15)
        elif len(Sagittal_T1_files) > 15:
            Sagittal_T1_files = self.unpad_images_list(Sagittal_T1_files, 15)
        for file in Sagittal_T1_files:
            original_dicom = pydicom.dcmread(os.path.join(Sagittal_T1_path, file)).pixel_array
            original_dicom = (original_dicom - original_dicom.min()) / (original_dicom.max() - original_dicom.min() + 1e-6) * 255
            resized_pixel_array = self.resize_image(original_dicom, (384, 384))

            axial_stack[..., k] = resized_pixel_array 
            k += 1 

        df_classes = self.divide_Axiel(sub_set)
        l = df_classes['path'].loc[(df_classes['class_id'] == category) | (df_classes['class_id'] == two_classes_category)].unique()

        l = l.tolist()
        l = sorted(l, key=self.extract_number)

        if len(l) < 10:
            l = self.pad_images_list(l, 10)
        elif len(l) > 10:
            l = self.unpad_images_list(l, 10)
        
        for file in l:
            original_dicom = pydicom.dcmread(file).pixel_array
            original_dicom = (original_dicom - original_dicom.min()) / (original_dicom.max() - original_dicom.min() + 1e-6) * 255
            axial_stack[..., k] = self.resize_image(original_dicom, (384, 384))
            k += 1
        
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
        sagittal_l1_l2 = np.zeros((192, 192, 15), dtype = np.uint8)
        axial_l1_l2 = np.zeros((384, 384, 25), dtype = np.uint8)
        sagittal_l2_l3 = np.zeros((192, 192, 15), dtype = np.uint8)
        axial_l2_l3 = np.zeros((384, 384, 25), dtype = np.uint8)
        sagittal_l3_l4 = np.zeros((192, 192, 15), dtype = np.uint8)
        axial_l3_l4 = np.zeros((384, 384, 25), dtype = np.uint8)
        sagittal_l4_l5 = np.zeros((192, 192, 15), dtype = np.uint8)
        axial_l4_l5 = np.zeros((384, 384, 25), dtype = np.uint8)
        sagittal_l5_s1 = np.zeros((192, 192, 15), dtype = np.uint8)
        axial_l5_s1 = np.zeros((384, 384, 25), dtype = np.uint8)
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
            
            elif col == "general_path_to_Sagittal_T2":
                Sagittal_T2_path = sub_set[col].iloc[0]
                Sagittal_T2_files = os.listdir(Sagittal_T2_path)
                Sagittal_T2_files = sorted(Sagittal_T2_files, key=self.extract_number)
                middle_index = len(Sagittal_T2_files) // 2
                Sagittal_T2_bboxes = cfg.segmentation.inference(os.path.join(Sagittal_T2_path, Sagittal_T2_files[middle_index]))
                if _count_neg_ones(Sagittal_T2_bboxes) != 0:
                    for p in range(len(Sagittal_T2_files)):
                        temp = cfg.segmentation.inference(os.path.join(Sagittal_T2_path, Sagittal_T2_files[p]))
                        for key, value in temp.items():
                            if value != [(-1, -1, -1, -1)]:
                                if key not in Sagittal_T2_bboxes or Sagittal_T2_bboxes[key] == [(-1, -1, -1, -1)]:
                                    Sagittal_T2_bboxes[key] = value
                        if _count_neg_ones(Sagittal_T2_bboxes) == 0:
                            break

        decription_df = self.df_description[(self.df_description['study_id'] == study_id)]
        sagittal_l1_l2, axial_l1_l2 = self.create_stack(sagittal_l1_l2, axial_l1_l2, Sagittal_T1_files, Sagittal_T1_path, Sagittal_T2_files, Sagittal_T2_bboxes,
                                        Sagittal_T2_path,
                                        "L1", "L1-L2", decription_df)
        
        sagittal_l2_l3, axial_l2_l3 = self.create_stack(sagittal_l2_l3, axial_l2_l3, Sagittal_T1_files, Sagittal_T1_path, Sagittal_T2_files, Sagittal_T2_bboxes,
                                        Sagittal_T2_path,
                                        "L2", "L2-L3", decription_df)
        
        sagittal_l3_l4, axial_l3_l4 = self.create_stack(sagittal_l3_l4, axial_l3_l4, Sagittal_T1_files, Sagittal_T1_path, Sagittal_T2_files, Sagittal_T2_bboxes,
                                        Sagittal_T2_path,
                                        "L3", "L3-L4", decription_df)
        
        sagittal_l4_l5, axial_l4_l5 = self.create_stack(sagittal_l4_l5, axial_l4_l5, Sagittal_T1_files, Sagittal_T1_path, Sagittal_T2_files, Sagittal_T2_bboxes,
                                        Sagittal_T2_path,
                                        "L4", "L4-L5", decription_df)
        
        sagittal_l5_s1, axial_l5_s1 = self.create_stack(sagittal_l5_s1, axial_l5_s1, Sagittal_T1_files, Sagittal_T1_path, Sagittal_T2_files, Sagittal_T2_bboxes,
                                        Sagittal_T2_path,
                                        "L5", "L5-S1", decription_df)
        # self.plot(sagittal_l1_l2[...,])
        # self.plot(sagittal_l2_l3[...,])
        # self.plot(sagittal_l3_l4[...,])
        # self.plot(sagittal_l4_l5[...,])
        # self.plot(sagittal_l5_s1[...,])


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

        # self.plot(axial_l1_l2[...,])
        # self.plot(axial_l1_l2[...,])
        # self.plot(axial_l1_l2[...,])
        # self.plot(axial_l1_l2[...,])
        # self.plot(axial_l1_l2[...,])
        sagittal_l1_l2 = torch.tensor(sagittal_l1_l2).permute(2, 0, 1)
        sagittal_l2_l3 = torch.tensor(sagittal_l2_l3).permute(2, 0, 1)
        sagittal_l3_l4 = torch.tensor(sagittal_l3_l4).permute(2, 0, 1)
        sagittal_l4_l5 = torch.tensor(sagittal_l4_l5).permute(2, 0, 1)
        sagittal_l5_s1 = torch.tensor(sagittal_l5_s1).permute(2, 0, 1)
        axial_l1_l2 = torch.tensor(axial_l1_l2).permute(2, 0, 1)
        axial_l2_l3 = torch.tensor(axial_l2_l3).permute(2, 0, 1)
        axial_l3_l4 = torch.tensor(axial_l3_l4).permute(2, 0, 1)
        axial_l4_l5 = torch.tensor(axial_l4_l5).permute(2, 0, 1)
        axial_l5_s1 = torch.tensor(axial_l5_s1).permute(2, 0, 1)
        


        t = self.df.iloc[index][1:]
        t = t.map(lambda y: cfg.label2id[y])
        labels = t.values.astype(np.int64)
        reordered_labels = self._reorder_of_labels(labels)
        return sagittal_l1_l2, sagittal_l2_l3, sagittal_l3_l4, sagittal_l4_l5, sagittal_l5_s1, axial_l1_l2, axial_l2_l3, axial_l3_l4, axial_l4_l5, axial_l5_s1, reordered_labels

def data_loader(train_data: Path, labels_path: Path, description_path: Path) -> tuple[DataLoader, DataLoader]:
    """
    Loads the data for training.
    Args:
        train_data (Path): The path to the training data.
        labels_path (Path): The path to the labels.
        description_path (Path): The path to the description.
    Returns:
        tuple[DataLoader, DataLoader]: A tuple containing two DataLoaders.
    """
    
    FAS_train = CustomDataset(train_data, labels_path, description_path, transforms_sagittal=transforms_sagittal, transforms_axial=transforms_axial)
    return FAS_train