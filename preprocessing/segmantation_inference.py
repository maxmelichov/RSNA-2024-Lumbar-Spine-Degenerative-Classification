import numpy as np 
from PIL import Image
from matplotlib.patches import Rectangle

import torch

from segmentation_models_pytorch import Unet

import numpy as np 
import os
from PIL import Image
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import pydicom

import albumentations as A
from albumentations.pytorch import ToTensorV2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

label_dict_clean = {1 : 'L5', 2 : 'L4', 3 : 'L3', 4 : 'L2', 5 : 'L1', 6 : 'T12',
                7 : 'unknown', 8 : 'unknown', 9 : 'unknown',
                10: 'spinal canal', 11: 'L5-S1', 12: 'L4-L5', 13: 'L3-L4',
                14: 'L2-L3', 15: 'L1-L2', 16: 'T12-L1',
                17: 'unknown', 18: 'unknown', 19: 'unknown'
             }
label_dict = {1 : '1: L5', 2 : '2: L4', 3 : '3: L3', 4 : '4: L2', 5 : '5: L1', 6 : '6: T12',
                7 : '7: unknown', 8 : '8: unknown', 9 : '9: unknown',
                10: '10: spinal canal', 11: '11: L5-S1', 12: '12: L4-L5', 13: '13: L3-L4',
                14: '14: L2-L3', 15: '15: L1-L2', 16: '16: T12-L1',
                17: '17: unknown', 18: '18: unknown', 19: '19: unknown'
             }


classes_of_interest = [1, 2, 3, 4, 5] # 1 is L5, 5 is L1, 11 is L5-S1, 15 is L1-L2

class SegmentaionInference:
    def __init__(self, model_path):
        #transforms
        self.newsize = (256, 256)

        self.num_classes = 20
        self.model_path = model_path
        self.load_model()
        self.transforms_valid = A.Compose([
            A.Resize(self.newsize[0], self.newsize[1]),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2()
        ])
    
    def load_model(self):
        self.model = Unet(
        encoder_name="resnet34",  # Choose encoder (e.g. resnet18, efficientnet-b0)
        classes=self.num_classes,  # Number of output classes
        in_channels=3  # Number of input channels (e.g. 3 for RGB)
        ).to(device)
        self.model.load_state_dict(torch.load("weights\simple_unet.pth"))
    
    def prepare_data(self, dcm_path):
        image = pydicom.dcmread(dcm_path).pixel_array
        image = Image.fromarray(image)
        if image.mode != 'RGB':  # Ensure image is RGB
            image = image.convert('RGB')
        image = np.asarray(image)
        if (image > 1).any():  # Normalize if pixel values are between 0-255
            image = image / 255.0
            
        self.original_image = image
        if self.transforms_valid is not None:
            transformed = self.transforms_valid(image=image)
            image = transformed["image"]
        image = torch.as_tensor(image).float()
        return image

    def predict_mask(self, dcm_path):
        self.model.eval()
        image = self.prepare_data(dcm_path)
        with torch.no_grad():
            image = image.to(device)
            image = image.unsqueeze(0)
            outputs = self.model(image)
            pred = torch.argmax(outputs, dim=1)
        return pred.cpu()
    

    def get_class_bboxes(self, masks, classes_of_interest):
        def scale_bboxes(bboxes, original_size, segmented_size):
            orig_h, orig_w = original_size[0], original_size[1]
            seg_h, seg_w = segmented_size
            scaled_bboxes = {}
            
            for class_id, bbox_list in bboxes.items():
                scaled_bboxes[class_id] = []
                for bbox in bbox_list:
                    if bbox == (-1, -1, -1, -1):
                        scaled_bboxes[class_id].append(bbox)
                    else:
                        x, y, w, h = bbox
                        x = x * orig_w / seg_w
                        y = y * orig_h / seg_h
                        w = w * orig_w / seg_w
                        h = h * orig_h / seg_h
                        scaled_bboxes[class_id].append((x, y, w, h))
            
            return scaled_bboxes

        class_bboxes = {class_id: [] for class_id in classes_of_interest}
        
        for mask in masks:
            mask_np = mask.numpy()
            for class_id in classes_of_interest:
                coordinates = np.column_stack(np.where(mask_np == class_id))
                if coordinates.shape[0] > 0:
                    x_min, y_min = np.min(coordinates, axis=0)
                    x_max, y_max = np.max(coordinates, axis=0)
                    width = x_max - x_min
                    height = y_max - y_min
                    class_bboxes[class_id].append((y_min, x_min, height, width)) # I swapped x and y to match the format of the bounding boxes
                else:
                    class_bboxes[class_id].append((-1, -1, -1, -1))  # Placeholder for classes not present in the mask
        scaled_bboxes = scale_bboxes(class_bboxes, self.original_image.shape , (256, 256))
        return scaled_bboxes
    
    def inference(self, dcm_path):
        pred = self.predict_mask(dcm_path)
        bboxes = self.get_class_bboxes(pred, classes_of_interest)
        return bboxes
    
    @staticmethod
    def visualize_dicom_with_labels(dicom_path, bboxes, classes_of_interest):
    
    
        image = pydicom.dcmread(dicom_path).pixel_array
        image = Image.fromarray(image)
        if image.mode != 'RGB':  # Ensure image is RGB
            image = image.convert('RGB')
        image = np.asarray(image)
        
        plt.figure(figsize=(5, 5))
        plt.imshow(image, cmap='gray')
        
        for class_id in classes_of_interest:
            bbox = bboxes[class_id][0]  # Assuming single image
            if bbox != (-1, -1, -1, -1):  # Check if the class is present
                x, y, w, h = bbox
                plt.gca().add_patch(Rectangle((x, y), w, h, linewidth=2, edgecolor='red', facecolor='none'))
                plt.scatter(x, y, color='blue', s=100, marker='x')
                plt.text(x, y, label_dict[class_id], color='white', fontsize=12, bbox=dict(facecolor='red', alpha=0.5))
        
        plt.title("DICOM Image with Bounding Boxes and Labels")
        plt.axis('off')
        plt.show()