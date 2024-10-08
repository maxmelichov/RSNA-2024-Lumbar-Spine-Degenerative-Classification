import glob
import os
import numpy as np
import pandas as pd
import pydicom
import cv2
from typing import Union
import sys
sys.path.insert(0, r'F:\Projects\Kaggle\RSNA-2024-Lumbar-Spine-Degenerative-Classification\preprocessing')
from detection_inference import DetectionInference, transforms

class CrossReferenceSagittal:
    def __init__(self, image_dir :str = r"train_images\\", detection_model_path: str = r"weights\axial_detection_resnet18.pth"):
        self.image_dir = image_dir
        self.detection = DetectionInference(model_path=detection_model_path, transforms=transforms)

    def _convert_to_8bit(x: np.ndarray) -> np.ndarray:
        lower, upper = np.percentile(x, (1, 99))
        x = np.clip(x, lower, upper)
        x = x - np.min(x)
        x = x / np.max(x) 
        return (x * 255).astype("uint8")
    
    @staticmethod
    def _load_dicom_stack(dicom_folder: str, plane: str, reverse_sort: bool = False) -> dict:
        dicom_files = glob.glob(os.path.join(dicom_folder, "*.dcm"))
        dicoms = [pydicom.dcmread(f) for f in dicom_files]
        plane = {"sagittal": 0, "coronal": 1, "axial": 2}[plane.lower()]
        positions = np.asarray([float(d.ImagePositionPatient[plane]) for d in dicoms])
        # if reverse_sort=False, then increasing array index will be from RIGHT->LEFT and CAUDAL->CRANIAL
        # thus we do reverse_sort=True for axial so increasing array index is craniocaudal
        idx = np.argsort(-positions if reverse_sort else positions)
        ipp = np.asarray([d.ImagePositionPatient for d in dicoms]).astype("float")[idx]
        first_dicom = pydicom.dcmread(dicom_files[0])
        target_shape = first_dicom.pixel_array.shape
        array = np.stack([cv2.resize(d.pixel_array.astype("float32"), target_shape) for d in dicoms])
        array = array[idx]
        sorted_files = [dicom_files[i] for i in idx]
        return {"array": CrossReferenceSagittal._convert_to_8bit(array), "positions": ipp, "pixel_spacing": np.asarray(dicoms[0].PixelSpacing).astype("float"), "sorted_files": sorted_files}
    
    @staticmethod
    def _find_classes(x: int, min_x, max_x) -> list:

        def get_left_center_right(min_x, max_x):
            
            margin = (max_x - min_x) // 3
            # Define the size of the crop
            left_x_min = min_x 
            left_x_max = min_x + margin
            center_x_min = min_x + margin
            center_x_max = min_x + margin * 2
            right_x_min = min_x + margin
            right_x_max = max_x
            return (left_x_min, left_x_max), (center_x_min, center_x_max), (right_x_min, right_x_max)
            
            
        
        left, center, right = get_left_center_right(min_x, max_x)
        def categorize(x, left, center, right):
            in_left = left[0] <= x <= left[1]
            in_center = center[0] <= x <= center[1]
            in_right = right[0] <= x <= right[1]

            if in_left and in_center:
                return "left-center"
            elif in_right and in_center:
                return "right-center"
            elif in_left:
                return "left"
            elif in_center:
                return "center"
            elif in_right:
                return "right"
            return -1  # or perhaps return "out-of-range" or similar to maintain type consistency
        # print(categorize(x, left, center, right), x, left, center, right)
        return categorize(x, left, center, right)

        

    def _infer_axis_from_study_for_test(self, sag_t: dict, ax_t2: dict) -> pd.DataFrame:
        classes_df = pd.DataFrame(columns=["path", "side"])
        top_left_hand_corner_ax_t2 = ax_t2["positions"][len(ax_t2["array"]) // 2]
        ax_x_axis_to_pixel_space = [top_left_hand_corner_ax_t2[0]]
        
        while len(ax_x_axis_to_pixel_space) < ax_t2["array"].shape[2]: 
            ax_x_axis_to_pixel_space.append(ax_x_axis_to_pixel_space[-1] + ax_t2["pixel_spacing"][0])
            
        ax_x_coord_to_sag_slice = {}
        for sag_t2_slice, sag_t2_pos in zip(sag_t["array"], sag_t["positions"]):
            diffs = np.abs(np.asarray(ax_x_axis_to_pixel_space) - sag_t2_pos[0])
            ax_x_coord = np.argmin(diffs)
            ax_x_coord_to_sag_slice[ax_x_coord] = sag_t2_slice

            original_dicom = pydicom.dcmread(sag_t["sorted_files"][len(sag_t["sorted_files"]) // 2]).pixel_array
            original_dicom = original_dicom.clip(np.percentile(original_dicom, 1), np.percentile(original_dicom, 99))
            # bboxes = self.detection.inference(original_dicom, None, None)
        min_x = min([*ax_x_coord_to_sag_slice])
        max_x = max([*ax_x_coord_to_sag_slice])
        for i, x in enumerate([*ax_x_coord_to_sag_slice]):
            side = CrossReferenceSagittal._find_classes(x, min_x, max_x)
            if side != -1:
                classes_df.loc[len(classes_df)] = [sag_t["sorted_files"][i], side]
                
        return classes_df

    def get_cross_reference_for_Sagittal(self, study: pd.DataFrame, mode = "train", type = "t1") -> Union[None, pd.DataFrame]:
            sag_t1, sag_t2, ax_t2 = None, None, None
            for row in study.itertuples():
                if row.series_description == "Sagittal T2/STIR":
                    sag_t2 = CrossReferenceSagittal._load_dicom_stack(os.path.join(self.image_dir, str(row.study_id), str(row.series_id)), plane="sagittal")
                elif row.series_description == "Sagittal T1":
                    sag_t1 = CrossReferenceSagittal._load_dicom_stack(os.path.join(self.image_dir, str(row.study_id), str(row.series_id)), plane="sagittal")
                elif row.series_description == "Axial T2":
                    ax_t2 = CrossReferenceSagittal._load_dicom_stack(os.path.join(self.image_dir, str(row.study_id), str(row.series_id)), plane="axial", reverse_sort=True)
            if mode == "train":
                # if sag_t2 and ax_t2:
                #     self._infer_axis_from_study(sag_t2, ax_t2)
                # elif sag_t1 and ax_t2:
                #     self._infer_axis_from_study(sag_t1, ax_t2)
                # else:
                #     print("Could not find Sagittal T2/STIR or Sagittal T1 and Axial T2 for this study. Study ID:", study.study_id)
                pass
            else:
                if type == "t2":
                    return self._infer_axis_from_study_for_test(sag_t2, ax_t2)
                elif type == "t1":
                    return self._infer_axis_from_study_for_test(sag_t1, ax_t2)
                else:
                    print("Could not find Sagittal T2/STIR or Sagittal T1 and Axial T2 for this study. Study ID:", study.study_id)