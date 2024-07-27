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
segmentation = SegmentaionInference(model_path=r"weights\simple_unet.pth")
label2id = {'Normal/Mild': 0, 'Moderate':1, 'Severe':2}
category2id = {"L1": 0, "L2": 1, "L3": 2, "L4": 3, "L5": 4}
skip_study_id = [2492114990, 2780132468, 3008676218]

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
    def __init__(self, data_path,labels_path, transform):
        self.df = pd.read_csv(data_path)
        self.df_labels = pd.read_csv(labels_path)
        self.transform = transform
        pass
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
    def center_crop(dcm_path, bboxes): # need to check this function
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
        return scipy.ndimage.zoom(pixel_array, (new_size[0]/pixel_array.shape[0], new_size[1]/pixel_array.shape[1]), order=3)  # order=3 for bicubic
        # if pixel_array.shape == new_size:
        #     return pixel_array
        # elif pixel_array.shape[0] > new_size[0] or pixel_array.shape[1] > new_size[1]:
        #     return scipy.ndimage.zoom(pixel_array, (new_size[0]/pixel_array.shape[0], new_size[1]/pixel_array.shape[1]), order=3)
        # else:
        #     image = Image.fromarray(pixel_array)
        #     return image.resize((512, 512), Image.LANCZOS)  # You can use Image.LANCZOS for potentially better quality


    def __getitem__(self, index):
        Axial_T2 = np.zeros((512, 512, 10), dtype = np.uint8)
        Sagittal_T1 = np.zeros((256, 256, 15), dtype = np.uint8)
        Sagittal_T2_STIR = np.zeros((256, 256, 15), dtype = np.uint8)
        study_id = self.df_labels.iloc[index]['study_id']
        if study_id in skip_study_id:
            return self.__getitem__((index + 1) % len(self.df_labels))  # Try next item, wrap around if at end
        category = self.df_labels.iloc[index]['category']
        secondary_category = self.df_labels.iloc[index]['secondary_category']
        sub_set = self.df[(self.df['study_id'] == study_id) & ((self.df['category'] == category) | (self.df['category'] == secondary_category))]
        if len(sub_set) == 0:
            return self.__getitem__((index + 1) % len(self.df_labels))

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
                    # resize the image to 512x512
                    new_pixel_array = self.resize_image(dcm.pixel_array, (512, 512))
                    Axial_T2[..., i] = new_pixel_array

            elif col == "general_path_to_Sagittal_T1":
                path = sub_set[col].iloc[0]
                list_of_files = os.listdir(path)
                middle_index = len(list_of_files) // 2
                bboxes = segmentation.inference(os.path.join(path, list_of_files[middle_index]))
                if len(list_of_files) < 15:
                    list_of_files = self.pad_images_list(list_of_files, 15)
                elif len(list_of_files) > 15:
                    list_of_files = self.unpad_images_list(list_of_files, 15)
                for i, file in enumerate(list_of_files):
                    new_pixel_array = self.center_crop(os.path.join(path, file), bboxes)
                    # resize the image to 256x256
                    resized_pixel_array = self.resize_image(new_pixel_array, (256, 256))
                    Sagittal_T1[..., i] = resized_pixel_array
                    

            elif col == "general_path_to_Sagittal_T2_STIR":
                path = sub_set[col].iloc[0]
                list_of_files = os.listdir(path)
                middle_index = len(list_of_files) // 2
                bboxes = segmentation.inference(os.path.join(path, list_of_files[middle_index]))
                if len(list_of_files) < 15:
                    list_of_files = self.pad_images_list(list_of_files, 15)
                elif len(list_of_files) > 15:
                    list_of_files = self.unpad_images_list(list_of_files, 15)
                for i, file in enumerate(list_of_files):
                    new_pixel_array = self.center_crop(os.path.join(path, file), bboxes)
                    # resize the image to 256x256
                    resized_pixel_array = self.resize_image(new_pixel_array, (256, 256))
                    Sagittal_T2_STIR[..., i] = resized_pixel_array
        
        if self.transform:
            Axial_T2 = self.transform(image=Axial_T2)['image']
            Sagittal_T1 = self.transform(image=Sagittal_T1)['image']
            Sagittal_T2_STIR = self.transform(image=Sagittal_T2_STIR)['image']
        
        Axial_T2 = torch.tensor(Axial_T2).permute(2, 0, 1)
        Sagittal_T1 = torch.tensor(Sagittal_T1).permute(2, 0, 1)
        Sagittal_T2_STIR = torch.tensor(Sagittal_T2_STIR).permute(2, 0, 1)

        category_hot = F.one_hot(torch.tensor(category2id[category]), num_classes=5)

        if type(self.df_labels.iloc[index]['spinal_canal_stenosis']) == float:
            label0 = [-100, -100, -100]
        else:
            label0 = F.one_hot(torch.tensor(label2id[self.df_labels.iloc[index]['spinal_canal_stenosis']]), num_classes=3)
        
        if type(self.df_labels.iloc[index]['left_neural_foraminal_narrowing']) == float:
            label1 =[-100, -100, -100]
        else:
            label1 = F.one_hot(torch.tensor(label2id[self.df_labels.iloc[index]['left_neural_foraminal_narrowing']]), num_classes=3)
        
        if type(self.df_labels.iloc[index]['right_neural_foraminal_narrowing']) == float:
            label2 = [-100, -100, -100]
        else:
            label2 = F.one_hot(torch.tensor(label2id[self.df_labels.iloc[index]['right_neural_foraminal_narrowing']]), num_classes=3)

        if type(self.df_labels.iloc[index]['left_subarticular_stenosis']) == float:
            label3 = [-100, -100, -100]
        else:
            label3 = F.one_hot(torch.tensor(label2id[self.df_labels.iloc[index]['left_subarticular_stenosis']]), num_classes=3)
        
        if type(self.df_labels.iloc[index]['right_subarticular_stenosis']) == float:
            label4 = [-100, -100, -100]
        else:
            label4 = F.one_hot(torch.tensor(label2id[self.df_labels.iloc[index]['right_subarticular_stenosis']]), num_classes=3)
        
        flattened_list = [item for sublist in [label0, label1, label2, label3, label4] for item in sublist]

        # if type(self.df_labels.iloc[index]['spinal_canal_stenosis']) == float:
        #     label0 = -100
        # else:
        #     label0 = torch.tensor(label2id[self.df_labels.iloc[index]['spinal_canal_stenosis']])
        # if type(self.df_labels.iloc[index]['left_neural_foraminal_narrowing']) == float:
        #     label1 = -100
        # else:
        #     label1 = torch.tensor(label2id[self.df_labels.iloc[index]['left_neural_foraminal_narrowing']])
        # if type(self.df_labels.iloc[index]['right_neural_foraminal_narrowing']) == float:
        #     label2 = -100
        # else:
        #     label2 = torch.tensor(label2id[self.df_labels.iloc[index]['right_neural_foraminal_narrowing']])
        # if type(self.df_labels.iloc[index]['left_subarticular_stenosis']) == float:
        #     label3 = -100
        # else:
        #     label3 = torch.tensor(label2id[self.df_labels.iloc[index]['left_subarticular_stenosis']])
        # if type(self.df_labels.iloc[index]['right_subarticular_stenosis']) == float:
        #     label4 = -100
        # else:
        #     label4 = torch.tensor(label2id[self.df_labels.iloc[index]['right_subarticular_stenosis']])
        
        # labels = torch.tensor([label0, label1, label2, label3, label4], dtype=torch.float32)

        labels = torch.tensor(flattened_list, dtype=torch.float32)

        return Axial_T2, Sagittal_T1, Sagittal_T2_STIR, category_hot, labels


def data_loader(train_data: Path, labels_path: Path) -> tuple[DataLoader, DataLoader]:
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
    FAS_train = CustomDataset(train_data, labels_path, transforms_train)
    # FAS_val = CustomDataset(dir_csv_val, transforms_val, ela)
    
    # train_loader = DataLoader(FAS_train, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(FAS_val, batch_size=batch_size, shuffle=True)
    return FAS_train