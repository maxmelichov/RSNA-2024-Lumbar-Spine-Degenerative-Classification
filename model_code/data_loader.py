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
segmentation = SegmentaionInference(model_path=r"weights\simple_unet.pth")
label2id = {'Normal/Mild': 0, 'Moderate':1, 'Severe':2}

AUG_PROB = 0.75
IMG_SIZE = [512, 512]
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
    A.Resize(IMG_SIZE[0], IMG_SIZE[1]),
    A.CoarseDropout(max_holes=16, max_height=64, max_width=64, min_holes=1, min_height=8, min_width=8, p=AUG_PROB),    
    A.Normalize(mean=0.5, std=0.5)
])

transforms_val = A.Compose([
    A.Resize(IMG_SIZE[0], IMG_SIZE[1]),
    A.Normalize(mean=0.5, std=0.5)
])

if not NOT_DEBUG or not AUG:
    transforms_train = transforms_val


class CustomDataset(Dataset):
    def __init__(self, data_path,labels_path, transform=None):
        self.df = pd.read_csv(data_path)
        self.df_labels = pd.read_csv(labels_path)
        pass
    def __len__(self):
        return len(self.df_labels)

    @staticmethod
    def pad_images_list(images_list, max_len):
        if len(images_list) < 0:
            raise ValueError("images_list is empty")
        if len(images_list) == max_len:
            return images_list
        
        current_length = len(images_list)
        if current_length >= max_len:
            return images_list[:max_len]
        
        # Calculate the number of duplicates needed
        duplicates_needed = images_list - current_length
        result = []
        center_index = current_length // 2
        
        # Add elements to the result, focusing on duplicating center elements
        for i in range(current_length):
            # Calculate how many times the current element should be duplicated
            if i == center_index:
                duplicate_count = duplicates_needed // 2 + 1
            elif i == center_index - 1 and current_length % 2 == 0:
                duplicate_count = duplicates_needed // 2 + 1
            else:
                duplicate_count = 1
            
            result.extend([images_list[i]] * duplicate_count)
        
        # Trim the result to the target length if it exceeds
        return result[:max_len]


    @staticmethod
    def center_crop(dcm_path, bboxes):
        min_x, max_x = 999999, -1
        min_y, max_y = 999999, -1
        list_of_bboxes = [boxes[0] for boxes in bboxes.values()]
        for x, y, _, _ in list_of_bboxes:
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)
        original_dicom = pydicom.dcmread(dcm_path)
        pixel_array = original_dicom.pixel_array
        
        # Convert DICOM pixel array to PIL image
        image = Image.fromarray(pixel_array)

        # Crop the image using the calculated bounding box
        cropped_image_array = np.array(image.crop((min_x, min_y, max_x + 1, max_y + 1)))

        # Ensure the data type matches the original DICOM's pixel array
        cropped_image_array = cropped_image_array.astype(pixel_array.dtype)
        return cropped_image_array

    @staticmethod
    def unpad_images_list(images_list, max_len):
        i = 0
        while len(images_list) > max_len:
            if i % 2 == 0:
                images_list.pop(-1)
            else:
                images_list.pop(0)
            i += 1



    def __getitem__(self, index):
        Axial_T2 = np.zeros((512, 512, 10), dtype = np.uint8)
        Sagittal_T1 = np.zeros((256, 256, 15), dtype = np.uint8)
        Sagittal_T2_STIR = np.zeros((256, 256, 15), dtype = np.uint8)
        study_id = self.df_labels.iloc[index]['study_id']
        category = self.df_labels.iloc[index]['category']
        secondary_category = self.df_labels.iloc[index]['secondary_category']
        sub_set = self.df[self.df['study_id'] == study_id, self.df['category'] == category or self.df['category'] == secondary_category]

        for col in sub_set.columns:
            if col == "general_path_to_Axial":
                path = sub_set[col].iloc[0]
                list_of_files = os.listdir(path)
                if len(list_of_files) < 10:
                    list_of_files = self.pad_images_list(list_of_files, 10)
                elif len(list_of_files) > 10:
                    list_of_files = self.unpad_images_list(list_of_files, 10)
                for i, file in enumerate(list_of_files):
                    dcm = pydicom.dcmread(os.path.join(path, file))
                    Axial_T2[..., i] = dcm.pixel_array
            elif col == "general_path_to_Sagittal_T1":
                path = sub_set[col].iloc[0]
                list_of_files = os.listdir(path)
                middle_index = len(list_of_files) // 2
                bboxes = segmentation.inference(list_of_files[middle_index])
                if len(list_of_files) < 15:
                    list_of_files = self.pad_images_list(list_of_files, 15)
                elif len(list_of_files) > 15:
                    list_of_files = self.unpad_images_list(list_of_files, 15)

                for i, file in enumerate(list_of_files):
                    self.center_crop(os.path.join(path, file), bboxes)
                    Sagittal_T1[..., i] = dcm.pixel_array
            elif col == "general_path_to_Sagittal_T2_STIR":
                path = sub_set[col].iloc[0]
                list_of_files = os.listdir(path)
                middle_index = len(list_of_files) // 2
                bboxes = segmentation.inference(list_of_files[middle_index])
                if len(list_of_files) < 15:
                    list_of_files = self.pad_images_list(list_of_files, 15)
                elif len(list_of_files) > 15:
                    list_of_files = self.unpad_images_list(list_of_files, 15)

                for i, file in enumerate(list_of_files):
                    self.center_crop(os.path.join(path, file), bboxes)
                    Sagittal_T2_STIR[..., i] = dcm.pixel_array
        
        if self.transform:
            Axial_T2 = self.transform(image=Axial_T2)['image']
            Sagittal_T1 = self.transform(image=Sagittal_T1)['image']
            Sagittal_T2_STIR = self.transform(image=Sagittal_T2_STIR)['image']
        
        Axial_T2 = torch.tensor(Axial_T2).transpose(2, 0, 1)
        Sagittal_T1 = torch.tensor(Sagittal_T1).transpose(2, 0, 1)
        Sagittal_T2_STIR = torch.tensor(Sagittal_T2_STIR).transpose(2, 0, 1)
        category_hot = F.one_hot(torch.tensor(category), num_classes=5)
        labels = torch.tensor([label2id[self.df_labels.iloc[index]['spinal_canal_stenosis']],
                              label2id[self.df_labels.iloc[index]['left_neural_foraminal_narrowing']],
                              label2id[self.df_labels.iloc[index]['right_neural_foraminal_narrowing']],
                              label2id[self.df_labels.iloc[index]['left_subarticular_stenosis']],
                                label2id[self.df_labels.iloc[index]['right_subarticular_stenosis']]], dtype=torch.float32)

        return Axial_T2, Sagittal_T1, Sagittal_T2_STIR, category_hot, labels
