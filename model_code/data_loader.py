import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import albumentations as A
import pydicom
import os
from PIL import Image
from preprocessing.segmantation_inference import SegmentaionInference, classes_of_interest, label_dict_clean
import torch.nn.functional as F
from pathlib import Path
import scipy.ndimage
import matplotlib.pyplot as plt
import re
import ast
from preprocessing.Cross_Reference import CrossReference
cross_reference = CrossReference()
segmentation = SegmentaionInference(model_path=r"weights\simple_unet.onnx")
label2id = {'Normal/Mild': 0, 'Moderate':1, 'Severe':2, np.nan: -100}
category2id = {"L1": 1, "L2": 2, "L3": 3, "L4": 4, "L5": 5, "L5-S1": 11, "L4-L5": 12, "L3-L4": 13, "L2-L3": 14, "L1-L2": 15}
skip_study_id = [2492114990, 2780132468, 3008676218]
two_classes_category = {11: 'L5-S1', 12: 'L4-L5', 13: 'L3-L4', 14: 'L2-L3', 15: 'L1-L2'}
AUG_PROB = 0.75
NOT_DEBUG = True
AUG = True

transforms_train = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=AUG_PROB),
    A.OneOf([
        A.MotionBlur(blur_limit=5),
        A.MedianBlur(blur_limit=5),
        A.GaussianBlur(blur_limit=5),
        A.GaussNoise(var_limit=(5.0, 30.0)),
    ], p=AUG_PROB),

    A.OneOf([
        A.OpticalDistortion(distort_limit=1.0),
        A.GridDistortion(num_steps=5, distort_limit=1.),
        A.ElasticTransform(alpha=3),
    ], p=AUG_PROB),

    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=AUG_PROB),
    A.CoarseDropout(max_holes=16, max_height=64, max_width=64, min_holes=1, min_height=8, min_width=8, p=AUG_PROB),    
    A.Normalize(mean=0.5, std=0.5)
])

transforms_val = A.Compose([
    A.Normalize(mean=0.5, std=0.5)
])

if not NOT_DEBUG or not AUG:
    transforms_train = transforms_val


class CustomDataset(Dataset):
    def __init__(self, train_data, labels_path, description_path, transform):
        self.df = pd.read_csv(train_data)
        self.df_labels = pd.read_csv(labels_path)
        self.df_description = pd.read_csv(description_path)
        self.transform = transform
        self.train_path = "train_images"
        pass
    def __len__(self):
        return len(self.df_labels)

    @staticmethod
    def plot(stack):
        fig, axes = plt.subplots(2, 5, figsize=(15, 9))
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


    # @staticmethod
    # def center_crop(dcm_path, bboxes): # need to check this function
    #     min_x, max_x = 999999, -1
    #     min_y, max_y = 999999, -1
    #     list_of_bboxes = [boxes[0] for boxes in bboxes.values()]
    #     for x, y, h, w in list_of_bboxes:
    #         if x == -1 and y == -1 and h == -1 and w == -1:
    #             continue
    #         min_x = min(min_x, x)
    #         max_x = max(max_x, x+w)
    #         min_y = min(min_y, y)
    #         max_y = max(max_y, y+h)
    #     if min_x == 999999:
    #         min_x, min_y, max_x, max_y = 0, 0, 512, 512
    #     original_dicom = pydicom.dcmread(dcm_path)
    #     pixel_array = original_dicom.pixel_array
        
    #     # Convert DICOM pixel array to PIL image
    #     image = Image.fromarray(pixel_array)
    #     # Crop the image using the calculated bounding box
    #     cropped_image_array = np.array(image.crop((min_x, min_y, max_x + 1, max_y + 1)))
    #     return cropped_image_array
    
    @staticmethod
    def center_crop_by_category(original_dicom, bboxes, category, second_category): # need to check this function
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
        image = Image.fromarray(original_dicom)
        if (min_x == -1 and min_y == -1) or ((max_x - min_x < 35) or (max_y - min_y < 35)):
            # shape = np.array(original_dicom).shape
            # need to add plt.imshow() of the segmentation mask
            return original_dicom
        cropped_image = image.crop((min_x-10, min_y-10, max_x + 30, max_y))

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


    def create_stack(self, sagittal_stack, axial_stack, Sagittal_T1_files, Sagittal_T1_bboxes, Sagittal_T1_path,
                     Sagittal_T2_files, Sagittal_T2_bboxes, Sagittal_T2_path,
                     category, two_classes_category, sub_set):

        k = 0
        if len(Sagittal_T1_files) < 15:
            Sagittal_T1_files = self.pad_images_list(Sagittal_T1_files, 15)
        elif len(Sagittal_T1_files) > 15:
            Sagittal_T1_files = self.unpad_images_list(Sagittal_T1_files, 15)
        for file in Sagittal_T1_files:

            original_dicom = pydicom.dcmread(os.path.join(Sagittal_T1_path, file)).pixel_array
            original_dicom = (original_dicom - original_dicom.min()) / (original_dicom.max() - original_dicom.min() + 1e-6) * 255
            new_pixel_array = self.center_crop_by_category(original_dicom, Sagittal_T1_bboxes,
                                                           category2id[category], category2id[two_classes_category])
            resized_pixel_array = self.resize_image(new_pixel_array, (192, 192))
            # new_shape = new_pixel_array.shape
            # # Ensure the padded array is large enough to hold new_pixel_array
            # padded_array = np.zeros((max(224, new_shape[0]), max(128, new_shape[1])))

            # # Compute the starting indices for centering the new_pixel_array
            # start_x = (padded_array.shape[0] - new_shape[0]) // 2
            # start_y = (padded_array.shape[1] - new_shape[1]) // 2

            # Place the new_pixel_array in the center of the padded_array
            # padded_array[start_x:start_x + new_shape[0], start_y:start_y + new_shape[1]] = new_pixel_array
            sagittal_stack[..., k] = resized_pixel_array 
            k += 1
        
        if len(Sagittal_T2_files) < 15:
            Sagittal_T2_files = self.pad_images_list(Sagittal_T2_files, 15)
        elif len(Sagittal_T2_files) > 15:
            Sagittal_T2_files = self.unpad_images_list(Sagittal_T2_files, 15)
        for file in Sagittal_T2_files:
            original_dicom = pydicom.dcmread(os.path.join(Sagittal_T2_path, file)).pixel_array
            original_dicom = (original_dicom - original_dicom.min()) / (original_dicom.max() - original_dicom.min() + 1e-6) * 255
            new_pixel_array = self.center_crop_by_category(original_dicom, Sagittal_T2_bboxes,
                                                           category2id[category], category2id[two_classes_category])
            resized_pixel_array = self.resize_image(new_pixel_array, (192, 192))

            # # Get the shape of the new_pixel_array
            # new_shape = new_pixel_array.shape
            # # Ensure the padded array is large enough to hold new_pixel_array
            # padded_array = np.zeros((max(224, new_shape[0]), max(128, new_shape[1])))

            # # Compute the starting indices for centering the new_pixel_array
            # start_x = (padded_array.shape[0] - new_shape[0]) // 2
            # start_y = (padded_array.shape[1] - new_shape[1]) // 2

            # # Place the new_pixel_array in the center of the padded_array
            # padded_array[start_x:start_x + new_shape[0], start_y:start_y + new_shape[1]] = new_pixel_array
            sagittal_stack[..., k] = resized_pixel_array
            k += 1
        df_classes = self.divide_Axiel(sub_set)
        l = df_classes['path'].loc[(df_classes['class_id'] == category) | (df_classes['class_id'] == two_classes_category)].unique()

        l = l.tolist()
        l = sorted(l, key=self.extract_number)

        if len(l) < 10:
            l = self.pad_images_list(l, 10)
        elif len(l) > 10:
            l = self.unpad_images_list(l, 10)
        k = 0
        for file in l:
            original_dicom = pydicom.dcmread(file).pixel_array
            original_dicom = (original_dicom - original_dicom.min()) / (original_dicom.max() - original_dicom.min() + 1e-6) * 255
            axial_stack[..., k] = self.resize_image(original_dicom, (384, 384))
            k += 1
        
        return sagittal_stack, axial_stack

    @staticmethod
    def _is_dict_structure_correct(d):
        required_keys = {1, 2, 3, 4, 5, 11, 12, 13, 14, 15}
        if set(d.keys()) != required_keys:
            return False
        
        for key in required_keys:
            if not (isinstance(d[key], list) and len(d[key]) == 1 and d[key][0] == (-1, -1, -1, -1)):
                return False
    
        return True
    
    @staticmethod
    def _is_all_black(image_array):
        return np.all(image_array == 0)

    def __getitem__(self, index):
        sagittal_l1_l2 = np.zeros((192, 192, 30), dtype = np.uint8)
        axial_l1_l2 = np.zeros((384, 384, 10), dtype = np.uint8)
        sagittal_l2_l3 = np.zeros((192, 192, 30), dtype = np.uint8)
        axial_l2_l3 = np.zeros((384, 384, 10), dtype = np.uint8)
        sagittal_l3_l4 = np.zeros((192, 192, 30), dtype = np.uint8)
        axial_l3_l4 = np.zeros((384, 384, 10), dtype = np.uint8)
        sagittal_l4_l5 = np.zeros((192, 192, 30), dtype = np.uint8)
        axial_l4_l5 = np.zeros((384, 384, 10), dtype = np.uint8)
        sagittal_l5_s1 = np.zeros((192, 192, 30), dtype = np.uint8)
        axial_l5_s1 = np.zeros((384, 384, 10), dtype = np.uint8)
        study_id = self.df_labels.iloc[index]['study_id']

        if study_id in skip_study_id:
            return self.__getitem__((index + 1) % len(self.df_labels))  # Try next item, wrap around if at end
    
        sub_set = self.df_labels[(self.df_labels['study_id'] == study_id)]

        def count_neg_ones(bboxes):
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
                Sagittal_T1_bboxes = segmentation.inference(os.path.join(Sagittal_T1_path, Sagittal_T1_files[middle_index]))
                if self._is_dict_structure_correct(Sagittal_T1_bboxes):
                    for p in range(middle_index - 5, middle_index + 6):
                        temp = segmentation.inference(os.path.join(Sagittal_T1_path, Sagittal_T1_files[p]))
                        if count_neg_ones(temp) < count_neg_ones(Sagittal_T1_bboxes):
                            Sagittal_T1_bboxes = temp
                        if not self._is_dict_structure_correct(Sagittal_T1_bboxes):
                            break
            elif col == "general_path_to_Sagittal_T2":
                Sagittal_T2_path = sub_set[col].iloc[0]
                Sagittal_T2_files = os.listdir(Sagittal_T2_path)
                Sagittal_T2_files = sorted(Sagittal_T2_files, key=self.extract_number)
                middle_index = len(Sagittal_T2_files) // 2
                Sagittal_T2_bboxes = segmentation.inference(os.path.join(Sagittal_T2_path, Sagittal_T2_files[middle_index]))
                if self._is_dict_structure_correct(Sagittal_T2_bboxes):
                    for p in range(middle_index - 5, middle_index + 6):
                        temp = segmentation.inference(os.path.join(Sagittal_T2_path, Sagittal_T2_files[p]))
                        if count_neg_ones(temp) < count_neg_ones(Sagittal_T2_bboxes):
                            Sagittal_T2_bboxes = temp
                        if not self._is_dict_structure_correct(Sagittal_T2_bboxes):
                            break
            # elif col == "general_path_to_Axial":
            #     Axial_path = sub_set[col].iloc[0]
            #     Axial_files = os.listdir(Axial_path)
            #     Axial_files = sorted(Axial_files, key=self.extract_number)

        decription_df = self.df_description[(self.df_description['study_id'] == study_id)]
        sagittal_l1_l2, axial_l1_l2 = self.create_stack(sagittal_l1_l2, axial_l1_l2, Sagittal_T1_files, Sagittal_T1_bboxes, Sagittal_T1_path, Sagittal_T2_files, Sagittal_T2_bboxes,
                                        Sagittal_T2_path,
                                        "L1", "L1-L2", decription_df)
        
        sagittal_l2_l3, axial_l2_l3 = self.create_stack(sagittal_l2_l3, axial_l2_l3, Sagittal_T1_files, Sagittal_T1_bboxes, Sagittal_T1_path, Sagittal_T2_files, Sagittal_T2_bboxes,
                                        Sagittal_T2_path,
                                        "L2", "L2-L3", decription_df)
        
        sagittal_l3_l4, axial_l3_l4 = self.create_stack(sagittal_l3_l4, axial_l3_l4, Sagittal_T1_files, Sagittal_T1_bboxes, Sagittal_T1_path, Sagittal_T2_files, Sagittal_T2_bboxes,
                                        Sagittal_T2_path,
                                        "L3", "L3-L4", decription_df)
        
        sagittal_l4_l5, axial_l4_l5 = self.create_stack(sagittal_l4_l5, axial_l4_l5, Sagittal_T1_files, Sagittal_T1_bboxes, Sagittal_T1_path, Sagittal_T2_files, Sagittal_T2_bboxes,
                                        Sagittal_T2_path,
                                        "L4", "L4-L5", decription_df)
        
        sagittal_l5_s1, axial_l5_s1 = self.create_stack(sagittal_l5_s1, axial_l5_s1, Sagittal_T1_files, Sagittal_T1_bboxes, Sagittal_T1_path, Sagittal_T2_files, Sagittal_T2_bboxes,
                                        Sagittal_T2_path,
                                        "L5", "L5-S1", decription_df)
        # if self._is_all_black(stack_l1_l2[..., 0]):
        #     print(study_id, Sagittal_T1_files, Sagittal_T2_files)
        #     print(Sagittal_T1_bboxes, Sagittal_T1_bboxes)
        # self.plot(sagittal_l1_l2[...,])
        # self.plot(sagittal_l2_l3[...,])
        # self.plot(sagittal_l3_l4[...,])
        # self.plot(sagittal_l4_l5[...,])
        # self.plot(sagittal_l5_s1[...,])


        if self.transform:
            sagittal_l1_l2 = self.transform(image=sagittal_l1_l2)['image']
            sagittal_l2_l3 = self.transform(image=sagittal_l2_l3)['image']
            sagittal_l3_l4 = self.transform(image=sagittal_l3_l4)['image']
            sagittal_l4_l5 = self.transform(image=sagittal_l4_l5)['image']
            sagittal_l5_s1 = self.transform(image=sagittal_l5_s1)['image']
            axial_l1_l2 = self.transform(image=axial_l1_l2)['image']
            axial_l2_l3 = self.transform(image=axial_l2_l3)['image']
            axial_l3_l4 = self.transform(image=axial_l3_l4)['image']
            axial_l4_l5 = self.transform(image=axial_l4_l5)['image']
            axial_l5_s1 = self.transform(image=axial_l5_s1)['image']


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
        t = t.map(lambda y: label2id[y])
        labels = t.values.astype(np.int64)

        return sagittal_l1_l2, sagittal_l2_l3, sagittal_l3_l4, sagittal_l4_l5, sagittal_l5_s1, axial_l1_l2, axial_l2_l3, axial_l3_l4, axial_l4_l5, axial_l5_s1, labels


class CustomDataset2(Dataset):
    def __init__(self, data_path,labels_path, transform):
        self.df = pd.read_csv(data_path)
        self.df_labels = pd.read_csv(labels_path)
        self.transform = transform


    def __len__(self):
        return len(self.df_labels)

    @staticmethod
    def pad_images_list(images_list, max_len): # need to check this function
        if len(images_list) < 0:
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
    def center_crop(original_dicom, bboxes): # need to check this function
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
        if min_x == 999999:
            min_x, min_y, max_x, max_y = 0, 0, 384, 384
        
        # Convert DICOM pixel array to PIL image
        image = Image.fromarray(original_dicom)
        # Crop the image using the calculated bounding box
        cropped_image_array = np.array(image.crop((min_x, min_y, max_x + 1, max_y + 1)))
        return cropped_image_array
    @staticmethod
    def crop_by_one_box(original_dicom, bbox):
        bbox = bbox[0]
        x, y, h, w = bbox
        if x == -1 and y == -1 and h == -1 and w == -1:
            return original_dicom
        elif h == 0 or w == 0:
            return original_dicom
        x, y, h, w = bbox
        image = Image.fromarray(original_dicom)
        cropped_image_array = np.array(image.crop((x-12, y-12, x+w+12, y+h+12 )))
        return cropped_image_array
    
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
        # return scipy.ndimage.zoom(pixel_array, (new_size[0]/pixel_array.shape[0], new_size[1]/pixel_array.shape[1]), order=3)  # order=3 for bicubic
        if pixel_array.shape == new_size:
            return pixel_array
        # elif pixel_array.shape[0] > new_size[0] or pixel_array.shape[1] > new_size[1]:
        #     return scipy.ndimage.zoom(pixel_array, (new_size[0]/pixel_array.shape[0], new_size[1]/pixel_array.shape[1]), order=3)
        else:
            # pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min() + 1e-6) * 255
            # image = Image.fromarray(pixel_array)
            return pixel_array.resize((new_size[0], new_size[1]))


    def __getitem__(self, index):
        saggital_l1_l2 = np.zeros((224, 224, 20), dtype = np.uint8)
        saggital_l2_l3 = np.zeros((224, 224, 20), dtype = np.uint8)
        saggital_l3_l4 = np.zeros((224, 224, 20), dtype = np.uint8)
        saggital_l4_l5 = np.zeros((224, 224, 20), dtype = np.uint8)
        saggital_l5_s1 = np.zeros((224, 224, 20), dtype = np.uint8)
        
        Axial_T2 = np.zeros((380, 380, 10), dtype = np.uint8)
        Sagittal_T1 = np.zeros((380, 380, 10), dtype = np.uint8)
        Sagittal_T2_STIR = np.zeros((380, 380, 10), dtype = np.uint8)
        x = np.zeros((380, 380, 30), dtype = np.uint8)
        study_id = self.df_labels.iloc[index]['study_id']
        if study_id in skip_study_id:
            return self.__getitem__((index + 1) % len(self.df_labels))  # Try next item, wrap around if at end
        sub_set = self.df[(self.df['study_id'] == study_id)]
        if len(sub_set) == 0:
            return self.__getitem__((index + 1) % len(self.df_labels))


        def extract_number(filename):
            match = re.search(r'\d+', filename)
            return int(match.group()) if match else 0
        k = 0
        for col in sub_set.columns:
            if col == "general_path_to_Sagittal_T1":
                path = sub_set[col].iloc[0]
                list_of_files = os.listdir(path)
                list_of_files = sorted(list_of_files, key=extract_number)
                middle_index = len(list_of_files) // 2
                bboxes = segmentation.inference(os.path.join(path, list_of_files[middle_index]))
                if len(list_of_files) < 10:
                    list_of_files = self.pad_images_list(list_of_files, 10)
                elif len(list_of_files) > 10:
                    list_of_files = self.unpad_images_list(list_of_files, 10)
                for i, file in enumerate(list_of_files):
                    original_dicom = pydicom.dcmread(os.path.join(path, file)).pixel_array
                    new_pixel_array = self.center_crop(original_dicom, bboxes)
                    # resize the image to 256x256
                    resized_pixel_array = self.resize_image(new_pixel_array, (380, 380))
                    Sagittal_T1[..., i] = resized_pixel_array

                    just_the_l1_l2 = self.crop_by_one_box(original_dicom, bboxes[15])
                    just_the_l1_l2 = self.resize_image(just_the_l1_l2, (224, 224))
                    saggital_l1_l2[..., k] = just_the_l1_l2
                    just_the_l2_l3 = self.crop_by_one_box(original_dicom, bboxes[14])
                    just_the_l2_l3 = self.resize_image(just_the_l2_l3, (224, 224))
                    saggital_l2_l3[..., k] = just_the_l2_l3
                    just_the_l3_l4 = self.crop_by_one_box(original_dicom, bboxes[13])
                    just_the_l3_l4 = self.resize_image(just_the_l3_l4, (224, 224))
                    saggital_l3_l4[..., k] = just_the_l3_l4
                    just_the_l4_l5 = self.crop_by_one_box(original_dicom, bboxes[12])
                    just_the_l4_l5 = self.resize_image(just_the_l4_l5, (224, 224))
                    saggital_l4_l5[..., k] = just_the_l4_l5
                    just_the_l5_s1 = self.crop_by_one_box(original_dicom, bboxes[11])
                    just_the_l5_s1 = self.resize_image(just_the_l5_s1, (224, 224))
                    saggital_l5_s1[..., k] = just_the_l5_s1
                    k+=1
       
            elif col == "general_path_to_Sagittal_T2":
                path = sub_set[col].iloc[0]
                list_of_files = os.listdir(path)
                list_of_files = sorted(list_of_files, key=extract_number)
                middle_index = len(list_of_files) // 2
                bboxes = segmentation.inference(os.path.join(path, list_of_files[middle_index]))
                # bboxes = self._get_bbox(index, "T2")
                if len(list_of_files) < 10:
                    list_of_files = self.pad_images_list(list_of_files, 10)
                elif len(list_of_files) > 10:
                    list_of_files = self.unpad_images_list(list_of_files, 10)
                for i, file in enumerate(list_of_files):
                    original_dicom = pydicom.dcmread(os.path.join(path, file)).pixel_array
                    new_pixel_array = self.center_crop(original_dicom, bboxes)
                    # resize the image to 256x256
                    resized_pixel_array = self.resize_image(new_pixel_array, (380, 380))
                    Sagittal_T2_STIR[..., i] = resized_pixel_array

                    just_the_l1_l2 = self.crop_by_one_box(original_dicom, bboxes[15])
                    just_the_l1_l2 = self.resize_image(just_the_l1_l2, (224, 224))
                    saggital_l1_l2[..., k] = just_the_l1_l2
                    just_the_l2_l3 = self.crop_by_one_box(original_dicom, bboxes[14])
                    just_the_l2_l3 = self.resize_image(just_the_l2_l3, (224, 224))
                    saggital_l2_l3[..., k] = just_the_l2_l3
                    just_the_l3_l4 = self.crop_by_one_box(original_dicom, bboxes[13])
                    just_the_l3_l4 = self.resize_image(just_the_l3_l4, (224, 224))
                    saggital_l3_l4[..., k] = just_the_l3_l4
                    just_the_l4_l5 = self.crop_by_one_box(original_dicom, bboxes[12])
                    just_the_l4_l5 = self.resize_image(just_the_l4_l5, (224, 224))
                    saggital_l4_l5[..., k] = just_the_l4_l5
                    just_the_l5_s1 = self.crop_by_one_box(original_dicom, bboxes[11])
                    just_the_l5_s1 = self.resize_image(just_the_l5_s1, (224, 224))
                    saggital_l5_s1[..., k] = just_the_l5_s1
                    k+=1
            

            elif col == "general_path_to_Axial":
                path = sub_set[col].iloc[0]
                list_of_files = os.listdir(path)
                axt2 = sorted(list_of_files, key=extract_number)
                step = len(axt2) / 10.0
                st = len(axt2)/2.0 - 4.0*step
                end = len(axt2)+0.0001
                for i, j in enumerate(np.arange(st, end, step)):
                    try:
                        p = axt2[max(0, int((j-0.5001).round()))]
                        dcm = pydicom.dcmread(os.path.join(path, p))
                        new_pixel_array = self.resize_image(dcm.pixel_array, (512, 512))
                        Axial_T2[..., i] = new_pixel_array
                    except:
                        pass 
              
        x = np.concatenate([Sagittal_T1, Sagittal_T2_STIR, Axial_T2], axis=2)
        if self.transform:
            x = self.transform(image=x)['image']
            saggital_l1_l2 = self.transform(image=saggital_l1_l2)['image']
            saggital_l2_l3 = self.transform(image=saggital_l2_l3)['image']
            saggital_l3_l4 = self.transform(image=saggital_l3_l4)['image']
            saggital_l4_l5 = self.transform(image=saggital_l4_l5)['image']
            saggital_l5_s1 = self.transform(image=saggital_l5_s1)['image']
        
        x = torch.tensor(x).permute(2, 0, 1)
        saggital_l1_l2 = torch.tensor(saggital_l1_l2).permute(2, 0, 1)
        saggital_l2_l3 = torch.tensor(saggital_l2_l3).permute(2, 0, 1)
        saggital_l3_l4 = torch.tensor(saggital_l3_l4).permute(2, 0, 1)
        saggital_l4_l5 = torch.tensor(saggital_l4_l5).permute(2, 0, 1)
        saggital_l5_s1 = torch.tensor(saggital_l5_s1).permute(2, 0, 1)

        t = self.df.iloc[index][1:]
        t = t.map(lambda x: label2id[x])
        labels = t.values.astype(np.int64)
        
        return x, saggital_l1_l2, saggital_l2_l3, saggital_l3_l4, saggital_l4_l5, saggital_l5_s1, labels

class CustomDataset3(Dataset):
    def __init__(self, data_path,labels_path, transform):
        self.df = pd.read_csv(data_path)
        self.df_labels = pd.read_csv(labels_path)
        self.transform = transform


    def __len__(self):
        return len(self.df_labels)

    @staticmethod
    def pad_images_list(images_list, max_len): # need to check this function
        if len(images_list) < 0:
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
    def center_crop(original_dicom, bboxes): # need to check this function
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
        if min_x == 999999:
            min_x, min_y, max_x, max_y = 0, 0, 380, 380
        
        # Convert DICOM pixel array to PIL image
        image = Image.fromarray(original_dicom)
        # Crop the image using the calculated bounding box
        cropped_image_array = np.array(image.crop((min_x, min_y, max_x + 1, max_y + 1)))
        return cropped_image_array
    @staticmethod
    def crop_by_one_box(original_dicom, bbox):
        bbox = bbox[0]
        x, y, h, w = bbox
        if x == -1 and y == -1 and h == -1 and w == -1:
            return original_dicom
        elif h == 0 or w == 0:
            return original_dicom
        x, y, h, w = bbox
        image = Image.fromarray(original_dicom)
        cropped_image_array = np.array(image.crop((x-12, y-12, x+w+12, y+h+12 )))
        return cropped_image_array
    
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
        # return scipy.ndimage.zoom(pixel_array, (new_size[0]/pixel_array.shape[0], new_size[1]/pixel_array.shape[1]), order=3)  # order=3 for bicubic
        if pixel_array.shape == new_size:
            return pixel_array
        # elif pixel_array.shape[0] > new_size[0] or pixel_array.shape[1] > new_size[1]:
        #     return scipy.ndimage.zoom(pixel_array, (new_size[0]/pixel_array.shape[0], new_size[1]/pixel_array.shape[1]), order=3)
        else:
            pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min() + 1e-6) * 255
            image = Image.fromarray(pixel_array)
            return image.resize((new_size[0], new_size[1]))


    def __getitem__(self, index):
        Axial_T2 = np.zeros((512, 512, 10), dtype = np.uint8)
        Sagittal_T1 = np.zeros((512, 512, 10), dtype = np.uint8)
        Sagittal_T2_STIR = np.zeros((512, 512, 10), dtype = np.uint8)
        x = np.zeros((512, 512, 30), dtype = np.uint8)
        study_id = self.df_labels.iloc[index]['study_id']
        if study_id in skip_study_id:
            return self.__getitem__((index + 1) % len(self.df_labels))  # Try next item, wrap around if at end
        sub_set = self.df[(self.df['study_id'] == study_id)]
        if len(sub_set) == 0:
            return self.__getitem__((index + 1) % len(self.df_labels))


        def extract_number(filename):
            match = re.search(r'\d+', filename)
            return int(match.group()) if match else 0
        k = 0
        for col in sub_set.columns:
            if col == "general_path_to_Sagittal_T1":
                path = sub_set[col].iloc[0]
                list_of_files = os.listdir(path)
                list_of_files = sorted(list_of_files, key=extract_number)
                # middle_index = len(list_of_files) // 2
                # bboxes = segmentation.inference(os.path.join(path, list_of_files[middle_index]))

                if len(list_of_files) < 10:
                    list_of_files = self.pad_images_list(list_of_files, 10)
                elif len(list_of_files) > 10:
                    list_of_files = self.unpad_images_list(list_of_files, 10)
                for i, file in enumerate(list_of_files):
                    original_dicom = pydicom.dcmread(os.path.join(path, file)).pixel_array
                    # new_pixel_array = self.center_crop(original_dicom, bboxes)
                    # resize the image to 256x256
                    resized_pixel_array = self.resize_image(original_dicom, (512, 512))
                    Sagittal_T1[..., i] = resized_pixel_array
       
            elif col == "general_path_to_Sagittal_T2":
                path = sub_set[col].iloc[0]
                list_of_files = os.listdir(path)
                list_of_files = sorted(list_of_files, key=extract_number)
                # middle_index = len(list_of_files) // 2
                # bboxes = segmentation.inference(os.path.join(path, list_of_files[middle_index]))
                # bboxes = self._get_bbox(index, "T2")
                if len(list_of_files) < 10:
                    list_of_files = self.pad_images_list(list_of_files, 10)
                elif len(list_of_files) > 10:
                    list_of_files = self.unpad_images_list(list_of_files, 10)
                for i, file in enumerate(list_of_files):
                    original_dicom = pydicom.dcmread(os.path.join(path, file)).pixel_array
                    # new_pixel_array = self.center_crop(original_dicom, bboxes)
                    # resize the image to 256x256
                    resized_pixel_array = self.resize_image(original_dicom, (512, 512))
                    Sagittal_T2_STIR[..., i] = resized_pixel_array
            

            elif col == "general_path_to_Axial":
                path = sub_set[col].iloc[0]
                list_of_files = os.listdir(path)
                axt2 = sorted(list_of_files, key=extract_number)
                step = len(axt2) / 10.0
                st = len(axt2)/2.0 - 4.0*step
                end = len(axt2)+0.0001
                for i, j in enumerate(np.arange(st, end, step)):
                    p = axt2[max(0, int((j-0.5001).round()))]
                    dcm = pydicom.dcmread(os.path.join(path, p))
                    new_pixel_array = self.resize_image(dcm.pixel_array, (512, 512))
                    Axial_T2[..., i] = new_pixel_array
              
        x = np.concatenate([Sagittal_T1, Sagittal_T2_STIR, Axial_T2], axis=2)
        if self.transform:
            x = self.transform(image=x)['image']

        
        x = torch.tensor(x).permute(2, 0, 1)

        t = self.df.iloc[index][1:]
        t = t.map(lambda y: label2id[y])
        labels = t.values.astype(np.int64)
        
        return x, labels

def data_loader(train_data: Path, labels_path: Path, description_path: Path) -> tuple[DataLoader, DataLoader]:
    """
    Loads and prepares the data for training and validation.

    Args:
        dir_csv_train (Path): The path to the training CSV file.
        dir_csv_val (Path): The path to the validation CSV file.
        batch_size (int): The batch size for training and validation.
        ela (bool, optional): Whether to apply Error Level Analysis (ELA) transformation. Defaults to False.

    Returns:
        tuple: A tuple containing the training data loader and the validation data loader.
    """
    FAS_train = CustomDataset(train_data, labels_path, description_path, transforms_train)
    
    # train_loader = DataLoader(FAS_train, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(FAS_val, batch_size=batch_size, shuffle=True)
    return FAS_train



