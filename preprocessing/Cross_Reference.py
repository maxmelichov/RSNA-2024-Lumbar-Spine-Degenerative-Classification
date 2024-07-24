import glob
import numpy as np
import os
import pydicom
from preprocessing.segmantation_inference import SegmentaionInference, label_dict_clean
import shutil
from skimage.transform import resize
from pathlib import Path
from PIL import Image
import pandas as pd


image_dir = "train_images/"
segmentation = SegmentaionInference(model_path=r"weights\simple_unet.pth")
# Cross-Reference Images in Different MRI Planes
class CrossReference:
    def __init__(self) -> None:
        pass

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
        arrays = []
        first_dicom = pydicom.dcmread(dicom_files[0])
        target_shape = first_dicom.pixel_array.shape
        for d in dicoms:
            pixel_array = d.pixel_array.astype("float32")
            if target_shape:
                pixel_array = resize(pixel_array, target_shape, mode='reflect', anti_aliasing=True)
            arrays.append(pixel_array)
        array = np.stack(arrays)[idx]
        sorted_files = [dicom_files[i] for i in idx]
        return {"array": CrossReference._convert_to_8bit(array), "positions": ipp, "pixel_spacing": np.asarray(dicoms[0].PixelSpacing).astype("float"), "sorted_files": sorted_files}
    
    @staticmethod
    def _find_classes(y: int, bbox: dict) -> list:
        classes = []
        for cls, boxes in bbox.items():
            if boxes == [(-1, -1, -1, -1)]:
                continue
            for box in boxes:
                _, y_min, _, height = box
                y_max = y_min + height
                if y_min <= y <= y_max:
                    classes.append(cls)
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

    @staticmethod
    def _split_by_class_and_save_Sagittal_relevent_parts(sag_t: dict, bboxes: dict, type_name: str) -> None:
        min_x, max_x = 999999, -1
        min_y, max_y = 999999, -1
        list_of_bboxes = [boxes[0] for boxes in bboxes.values()]
        for x, y, _, _ in list_of_bboxes:
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)
        print(min_x, max_x, min_y, max_y)

        for i in range(len(sag_t)):
            dcm_path = sag_t["sorted_files"][i]
            original_dicom = pydicom.dcmread(dcm_path)
            pixel_array = original_dicom.pixel_array
            
            # Convert DICOM pixel array to PIL image
            image = Image.fromarray(pixel_array)

            # Crop the image using the calculated bounding box
            cropped_image_array = np.array(image.crop((min_x, min_y, max_x + 1, max_y + 1)))

            # Ensure the data type matches the original DICOM's pixel array
            cropped_image_array = cropped_image_array.astype(pixel_array.dtype)

            # Encapsulate the pixel data if the original image was compressed
            # Check if the original DICOM uses a compressed transfer syntax
            if original_dicom.file_meta.TransferSyntaxUID.is_compressed:
                # Encapsulate the pixel data
                original_dicom.PixelData = pydicom.encaps.encapsulate([cropped_image_array.tobytes()])
            else:
                original_dicom.PixelData = cropped_image_array.tobytes()

            # Update the image dimensions to match the cropped size
            original_dicom.Rows, original_dicom.Columns = cropped_image_array.shape[:2]

            save_path, file_name = CrossReference._get_save_path_for_Sagittal(dcm_path, type_name)
            os.makedirs(save_path, exist_ok=True)

            # Save the modified DICOM file
            pydicom.filewriter.dcmwrite(os.path.join(save_path, file_name), original_dicom)
        

    @staticmethod
    def _infer_axis_from_study(sag_t: dict, ax_t2: dict) -> None:
        top_left_hand_corner_sag_t2 = sag_t["positions"][len(sag_t["array"]) // 2]
        sag_y_axis_to_pixel_space = [top_left_hand_corner_sag_t2[2]]
        while len(sag_y_axis_to_pixel_space) < sag_t["array"].shape[1]: 
            sag_y_axis_to_pixel_space.append(sag_y_axis_to_pixel_space[-1] - sag_t["pixel_spacing"][1])

        sag_y_coord_to_axial_slice = {}
        for ax_t2_slice, ax_t2_pos in zip(ax_t2["array"], ax_t2["positions"]):
            diffs = np.abs(np.asarray(sag_y_axis_to_pixel_space) - ax_t2_pos[2])
            sag_y_coord = np.argmin(diffs)
            sag_y_coord_to_axial_slice[sag_y_coord] = ax_t2_slice

        
        bboxes = segmentation.inference(sag_t["sorted_files"][len(sag_t["sorted_files"]) // 2])
        for i, y in enumerate([*sag_y_coord_to_axial_slice]):
            classes = CrossReference._find_classes(y, bboxes)
            if classes != -1:
                for cls in classes:
                    save_path, file_name = CrossReference._get_save_path_for_Axial(ax_t2["sorted_files"][i], cls)
                    os.makedirs(save_path, exist_ok=True)
                    shutil.copyfile(ax_t2["sorted_files"][i], os.path.join(save_path, file_name))


    @staticmethod
    def get_cross_reference_for_Axial(study: pd.DataFrame) -> None:
        image_dir = "train_images\\"
        sag_t1, sag_t2, ax_t2 = None, None, None
        for row in study.itertuples():
            if row.series_description == "Sagittal T2/STIR":
                sag_t2 = CrossReference._load_dicom_stack(os.path.join(image_dir, str(row.study_id), str(row.series_id)), plane="sagittal")
            elif row.series_description == "Sagittal T1":
                sag_t1 = CrossReference._load_dicom_stack(os.path.join(image_dir, str(row.study_id), str(row.series_id)), plane="sagittal")
            elif row.series_description == "Axial T2":
                ax_t2 = CrossReference._load_dicom_stack(os.path.join(image_dir, str(row.study_id), str(row.series_id)), plane="axial", reverse_sort=True)
        if sag_t2 and ax_t2:
            CrossReference._infer_axis_from_study(sag_t2, ax_t2)
        elif sag_t1 and ax_t2:
            CrossReference._infer_axis_from_study(sag_t1, ax_t2)
        else:
            print("Could not find Sagittal T2/STIR or Sagittal T1 and Axial T2 for this study. Study ID:", study.study_id)

        # bboxes_t1 = segmentation.inference(sag_t1["sorted_files"][len(sag_t1["sorted_files"]) // 2])
        # CrossReference._split_by_class_and_save_Sagittal_relevent_parts(sag_t1, bboxes_t1, "T1")

        # bboxes_t2 = segmentation.inference(sag_t2["sorted_files"][len(sag_t2["sorted_files"]) // 2])
        # CrossReference._split_by_class_and_save_Sagittal_relevent_parts(sag_t2, bboxes_t2, "T2_STIR")
        
    