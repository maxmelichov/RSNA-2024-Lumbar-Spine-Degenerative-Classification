import glob
import numpy as np
import os
import pydicom
import sys
sys.path.insert(0, r'F:\Projects\Kaggle\RSNA-2024-Lumbar-Spine-Degenerative-Classification\preprocessing')
from segmantation_inference import SegmentaionInference, label_dict_clean
import shutil
import cv2
from pathlib import Path
from PIL import Image
import pandas as pd
from typing import Union
import matplotlib.pyplot as plt



# Cross-Reference Images in Different MRI Planes
class CrossReferenceAxial:
    def __init__(self, image_dir :str = r"train_images\\", segmentation_model_path: str = r"weights\simple_unet.onnx"):
        self.image_dir = image_dir
        self.segmentation = SegmentaionInference(segmentation_model_path)
    @staticmethod
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
        iop = np.asarray([d.ImageOrientationPatient for d in dicoms]).astype("float")[idx]
        first_dicom = pydicom.dcmread(dicom_files[0])
        target_shape = first_dicom.pixel_array.shape
        target_shape = (target_shape[1], target_shape[0])
        array = np.stack([cv2.resize(d.pixel_array.astype("float32"), target_shape) for d in dicoms])
        array = array[idx]
        sorted_files = [dicom_files[i] for i in idx]
        return {"array": CrossReferenceAxial._convert_to_8bit(array), "positions": ipp, "orientations": iop, "pixel_spacing": np.asarray(dicoms[0].PixelSpacing).astype("float"), "sorted_files": sorted_files}
    
    def is_line_through_plane(x1, y1, x2, y2, x_min, y_min, width, height):
        # Check if the line is within the plane
        def point_to_line_distance(x, y, x1, y1, x2, y2):
            # Calculate the numerator of the distance formula
            numerator = abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1)
            
            # Calculate the denominator of the distance formula
            denominator = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
            
            # Calculate the distance
            distance = numerator / denominator
            
            return distance

        def get_center_of_plane(x_min, y_min, width, height):
            # Calculate the center of the rectangle
            x_center = x_min + width / 2
            y_center = y_min + height / 2
            return x_center, y_center
        
        x_center, y_center = get_center_of_plane(x_min, y_min, width, height)

        # Calculate the distance from the line to the center of the rectangle
        distance = point_to_line_distance(x_center, y_center, x1, y1, x2, y2)
        if distance <= height:
            return True
        return False
    
    def _find_classes(x_1: float, y_1: float, x_2: float, y_2: float, bbox: dict) -> list:
        classes = []
        for cls, boxes in bbox.items():
            if boxes == [(-1, -1, -1, -1)]:
                continue
            for box in boxes:
                x_min, y_min, weight, height = box
                if CrossReferenceAxial.is_line_through_plane(x_1, y_1, x_2, y_2, x_min, y_min, weight, height):
                    classes.append(cls)
                    # print(y_min, y, y_max, box)
                    break  # If y is within this class, no need to check further boxes for this class
            
        return classes if classes else -1
    

    @staticmethod
    def _get_save_path_for_Axial(dcm_path: str, class_id: int) -> tuple:
        save_folder = Path("Axial_T2_Division_by_categories")
        parts = dcm_path.replace("\\", "/").split("/")
        study_id = parts[1]
        series_id = parts[2]
        file_name = parts[3]
        return os.path.join(save_folder, label_dict_clean[class_id], study_id, series_id), file_name
    

    @staticmethod
    def _get_save_path_for_Sagittal(dcm_path: str, type_name: str) -> tuple:
        if type_name == "T1":
            save_folder = Path("Sagittal_T1_Division_by_categories")
        else:
            save_folder = Path("Sagittal_T2_STIR_Division_by_categories")
        parts = dcm_path.replace("\\", "/").split("/")
        study_id = parts[1]
        series_id = parts[2]
        file_name = parts[3]
        return os.path.join(save_folder, study_id, series_id), file_name
    
    def point_to_plane_distance(point, plane_point, plane_normal):
        # Calculate the distance from the point to the plane
        point = np.array(point)
        plane_point = np.array(plane_point)
        plane_normal = np.array(plane_normal)
        
        d_plane = np.abs(np.dot(plane_normal, point - plane_point)) / np.linalg.norm(plane_normal)
        
        return d_plane

    def project_point_onto_plane(point, plane_point, plane_normal):
        # Project the point onto the plane
        point = np.array(point)
        plane_point = np.array(plane_point)
        plane_normal = np.array(plane_normal)
        
        d_plane = np.dot(plane_normal, point - plane_point) / np.linalg.norm(plane_normal)**2
        projected_point = point - d_plane * plane_normal
        
        return projected_point

    def point_in_square(projected_point, square_vertices):
        # Check if the projected point lies within the square
        v1 = square_vertices[1] - square_vertices[0]
        v2 = square_vertices[3] - square_vertices[0]
        
        vp = projected_point - square_vertices[0]
        
        dot11 = np.dot(v1, v1)
        dot12 = np.dot(v1, v2)
        dot22 = np.dot(v2, v2)
        dot1p = np.dot(v1, vp)
        dot2p = np.dot(v2, vp)
        
        inv_denom = 1 / (dot11 * dot22 - dot12 * dot12)
        u = (dot22 * dot1p - dot12 * dot2p) * inv_denom
        v = (dot11 * dot2p - dot12 * dot1p) * inv_denom
        
        return (u >= 0) and (v >= 0) and (u <= 1) and (v <= 1)

    def distance_point_to_segment(point, v1, v2):
        # Compute the distance from a point to a line segment
        v1, v2, point = np.array(v1), np.array(v2), np.array(point)
        segment_vector = v2 - v1
        point_vector = point - v1
        segment_length = np.linalg.norm(segment_vector)
        
        if segment_length == 0:
            return np.linalg.norm(point_vector)
        
        t = max(0, min(1, np.dot(point_vector, segment_vector) / (segment_length**2)))
        projection = v1 + t * segment_vector
        
        return np.linalg.norm(projection - point)

    def point_to_square_distance(self, point, square_vertices):
        # Ensure the vertices are numpy arrays
        square_vertices = [np.array(v) for v in square_vertices]
        
        # Compute the plane normal
        v1 = square_vertices[1] - square_vertices[0]
        v2 = square_vertices[3] - square_vertices[0]
        plane_normal = np.cross(v1, v2)
        
        # Calculate the distance to the plane
        d_plane = self.point_to_plane_distance(point, square_vertices[0], plane_normal)
        
        # Project the point onto the plane
        projected_point = self.project_point_onto_plane(point, square_vertices[0], plane_normal)
        
        # Check if the projected point is within the square
        if self.point_in_square(projected_point, square_vertices):
            return d_plane
        
        return np.inf
    
    def find_world_point_on_sag(self, sag_Oxyz, point_xyz, orient_xyz_xyz, axial_height, axial_width,
                            axial_sp_X, axial_sp_Y, sag_sp_X, sag_sp_Y):
        coord_diff = (point_xyz - sag_Oxyz)
        x_1 = np.round(coord_diff[1] / sag_sp_X).astype(int)
        y_1 = np.round(-1 * coord_diff[2] / sag_sp_Y).astype(int)

        slope_x, slope_y = orient_xyz_xyz[4:]
        slope_y = - slope_y

        new_point_coord = coord_diff + axial_height * axial_sp_Y * orient_xyz_xyz[3:]

        # this part doesn't work well
        x_2 = np.round(new_point_coord[1] / sag_sp_X).astype(int)
        y_2 = np.round(-1 * new_point_coord[2] / sag_sp_Y).astype(int)

        return x_1, y_1, x_2, y_2, slope_x, slope_y


    
    def _infer_axis_from_study(self, sag_t: dict, axs_t2: list[dict]) -> None:
        top_left_hand_corner_sag_t2 = sag_t["positions"][len(sag_t["array"]) // 2]
        sag_y_axis_to_pixel_space = [top_left_hand_corner_sag_t2[2]]
        while len(sag_y_axis_to_pixel_space) < sag_t["array"].shape[1]: 
            sag_y_axis_to_pixel_space.append(sag_y_axis_to_pixel_space[-1] - sag_t["pixel_spacing"][1])

        
        for ax_t2 in axs_t2:
            sag_y_coord_to_axial_slice = {}
            for ax_t2_slice, ax_t2_pos in zip(ax_t2["array"], ax_t2["positions"]):
                diffs = np.abs(np.asarray(sag_y_axis_to_pixel_space) - ax_t2_pos[2])
                sag_y_coord = np.argmin(diffs)
                sag_y_coord_to_axial_slice[sag_y_coord] = ax_t2_slice

            
            bboxes = self.segmentation.inference(sag_t["sorted_files"][len(sag_t["sorted_files"]) // 2])
            for i, y in enumerate([*sag_y_coord_to_axial_slice]):
                classes = CrossReferenceAxial._find_classes(y, bboxes)
                if classes != -1:
                    for cls in classes:
                        save_path, file_name = CrossReferenceAxial._get_save_path_for_Axial(ax_t2["sorted_files"][i], cls)
                        os.makedirs(save_path, exist_ok=True)
                        shutil.copyfile(ax_t2["sorted_files"][i], os.path.join(save_path, file_name))
    
    def _infer_axis_from_study_for_test(self, sag_t: dict, axs_t2: dict) -> pd.DataFrame:
        classes_df = pd.DataFrame(columns=["path", "class_id"])
        top_left_hand_corner_sag_t2 = sag_t["positions"][len(sag_t["array"]) // 2]
        sag_y_axis_to_pixel_space = [top_left_hand_corner_sag_t2[2]]
        
        while len(sag_y_axis_to_pixel_space) < sag_t["array"].shape[1]: 
            sag_y_axis_to_pixel_space.append(sag_y_axis_to_pixel_space[-1] - sag_t["pixel_spacing"][1])

        def _count_neg_ones(bboxes):
            count = 0
            for vals in bboxes.values():
                count += vals.count((-1, -1, -1, -1))
            return count
        
        for ax_t2 in axs_t2:
            sag_y_coord_to_axial_slice = {}
            for i, (ax_t2_pos, ax_t2_ori, ax_t2_slice) in enumerate(zip(ax_t2["positions"], ax_t2["orientations"], ax_t2["array"])): 
                line_ = self.find_world_point_on_sag(sag_t["positions"][len(sag_t["array"]) // 2], ax_t2_pos, ax_t2_ori, *ax_t2_slice.shape,
                                    *ax_t2["pixel_spacing"], *sag_t["pixel_spacing"])
                sag_y_coord_to_axial_slice[i] = line_

            bboxes = self.segmentation.inference(sag_t["sorted_files"][len(sag_t["sorted_files"]) // 2])
            if _count_neg_ones(bboxes) != 0:
                    pmin = 0
                    pmax = len(sag_t["sorted_files"])
                    for p in range(pmin, pmax):
                        temp = self.segmentation.inference(sag_t["sorted_files"][p])
                        for key, value in temp.items():
                            if value != [(-1, -1, -1, -1)]:
                                if key not in bboxes or bboxes[key] == [(-1, -1, -1, -1)]:
                                    bboxes[key] = value
                        if _count_neg_ones(bboxes) == 0:
                            break
            classes_df = pd.DataFrame(columns=["path", "class_id"])
            for i, (x_1, y_1, x_2, y_2, slope_x, slope_y) in sag_y_coord_to_axial_slice.items():
                classes = CrossReferenceAxial._find_classes(x_1, y_1, x_2, y_2, bboxes)
                if classes != -1:
                    for cls in classes:
                        classes_df.loc[len(classes_df)] = [ax_t2["sorted_files"][i], label_dict_clean[cls]]
                
        return classes_df


    
    def get_cross_reference_for_Axial(self, study: pd.DataFrame, mode = "train") -> Union[None, pd.DataFrame]:
        sag_t1, sag_t2, axs_t2 = None, None, []
        for row in study.itertuples():
            if row.series_description == "Sagittal T2/STIR":
                sag_t2 = CrossReferenceAxial._load_dicom_stack(os.path.join(self.image_dir, str(row.study_id), str(row.series_id)), plane="sagittal")
            elif row.series_description == "Sagittal T1":
                sag_t1 = CrossReferenceAxial._load_dicom_stack(os.path.join(self.image_dir, str(row.study_id), str(row.series_id)), plane="sagittal")
            elif row.series_description == "Axial T2":
                axs_t2.append(CrossReferenceAxial._load_dicom_stack(os.path.join(self.image_dir, str(row.study_id), str(row.series_id)), plane="axial", reverse_sort=True))
        if mode == "train":
            if sag_t2 and axs_t2:
                self._infer_axis_from_study(sag_t2, axs_t2)
            elif sag_t1 and axs_t2:
                self._infer_axis_from_study(sag_t1, axs_t2)
            else:
                print("Could not find Sagittal T2/STIR or Sagittal T1 and Axial T2 for this study. Study ID:", study.study_id)
        else:
            if sag_t2 and axs_t2:
                return self._infer_axis_from_study_for_test(sag_t2, axs_t2)
            elif sag_t1 and axs_t2:
                return self._infer_axis_from_study_for_test(sag_t1, axs_t2)
            
            else:
                print("Could not find Sagittal T2/STIR or Sagittal T1 and Axial T2 for this study. Study ID:", study.study_id)


if __name__ == "__main__":
    cross_reference = CrossReferenceAxial()
    df = pd.read_csv(r"F:\Projects\Kaggle\RSNA-2024-Lumbar-Spine-Degenerative-Classification\train_series_descriptions.csv")
    study = df.loc[df.study_id == 4003253]
    cross_reference.get_cross_reference_for_Axial(study, mode = "test")
    # cross_reference.get_cross_reference_for_Axial(study).to_csv("axial_cross_reference.csv", index=False)