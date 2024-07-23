import glob
import numpy as np
import os
import pydicom
from preprocessing.segmantation_inference import SegmentaionInference, label_dict_clean
import shutil
from skimage.transform import resize

image_dir = "train_images/"
segmentation = SegmentaionInference(model_path=r"weights\simple_unet.pth")
# Cross-Reference Images in Different MRI Planes
class CrossReference:
    def __init__(self) -> None:
        pass

    @staticmethod
    def _convert_to_8bit(x):
        lower, upper = np.percentile(x, (1, 99))
        x = np.clip(x, lower, upper)
        x = x - np.min(x)
        x = x / np.max(x) 
        return (x * 255).astype("uint8")

    @staticmethod
    def _load_dicom_stack(dicom_folder, plane, reverse_sort=False):
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
    def find_classes(y, bbox):
        classes = []
        for cls, boxes in bbox.items():
            if boxes == [(-1, -1, -1, -1)]:
                continue
            for box in boxes:
                x_min, y_min, width, height = box
                y_max = y_min + height
                if y_min <= y <= y_max:
                    classes.append(cls)
                    break  # If y is within this class, no need to check further boxes for this class
        return classes if classes else -1
    

    @staticmethod
    def _get_save_path(dcm_path, class_id):
        save_folder = os.path.join("division_by_categories_and_names")
        parts = dcm_path.replace("\\", "/").split("/")
        study_id = parts[1]
        series_id = parts[2]
        file_name = parts[3]
        return os.path.join(save_folder, label_dict_clean[class_id], study_id, series_id), file_name
    

    @staticmethod
    def infer_axis_from_study(study):
        image_dir = "train_images\\"
        for row in study.itertuples():
            if row.series_description == "Sagittal T2/STIR":
                sag_t2 = CrossReference._load_dicom_stack(os.path.join(image_dir, str(row.study_id), str(row.series_id)), plane="sagittal")
            elif row.series_description == "Axial T2":
                ax_t2 = CrossReference._load_dicom_stack(os.path.join(image_dir, str(row.study_id), str(row.series_id)), plane="axial", reverse_sort=True)
        if sag_t2 and ax_t2:
            top_left_hand_corner_sag_t2 = sag_t2["positions"][len(sag_t2["array"]) // 2]
            sag_y_axis_to_pixel_space = [top_left_hand_corner_sag_t2[2]]
            while len(sag_y_axis_to_pixel_space) < sag_t2["array"].shape[1]: 
                sag_y_axis_to_pixel_space.append(sag_y_axis_to_pixel_space[-1] - sag_t2["pixel_spacing"][1])

            sag_y_coord_to_axial_slice = {}
            for ax_t2_slice, ax_t2_pos in zip(ax_t2["array"], ax_t2["positions"]):
                diffs = np.abs(np.asarray(sag_y_axis_to_pixel_space) - ax_t2_pos[2])
                sag_y_coord = np.argmin(diffs)
                sag_y_coord_to_axial_slice[sag_y_coord] = ax_t2_slice

            
            bboxes = segmentation.inference(sag_t2["sorted_files"][len(sag_t2["sorted_files"]) // 2])
            for i, y in enumerate([*sag_y_coord_to_axial_slice]):
                classes = CrossReference.find_classes(y, bboxes)
                if classes != -1:
                    for cls in classes:
                        save_path, file_name = CrossReference._get_save_path(ax_t2["sorted_files"][i], cls)
                        os.makedirs(save_path, exist_ok=True)
                        shutil.copyfile(ax_t2["sorted_files"][i], os.path.join(save_path, file_name))
        else:
            print("Either Sagittal T2/STIR or Axial T2 is missing in the study: ", study.study_id.unique())
        
