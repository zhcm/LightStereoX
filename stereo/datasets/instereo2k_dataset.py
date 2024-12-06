import os
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from .dataset_template import DatasetTemplate


class InStereo2KDataset(DatasetTemplate):
    def __init__(self, data_root_path, split_file, augmentations, return_right_disp):
        super().__init__(data_root_path, split_file, augmentations)
        self.return_right_disp = return_right_disp

    def __getitem__(self, idx):
        item = self.data_list[idx]
        full_paths = [os.path.join(self.root, x) for x in item]
        left_path, right_path, left_disp_path = full_paths

        left_img = Image.open(left_path).convert('RGB')
        left_img = np.array(left_img, dtype=np.float32)

        right_img = Image.open(right_path).convert('RGB')
        right_img = np.array(right_img, dtype=np.float32)

        left_disp = Image.open(left_disp_path)
        left_disp = np.array(left_disp).astype(np.float32)
        occ_mask = left_disp == 0
        left_disp = left_disp/100

        sample = {
            'left': left_img,
            'right': right_img,
            'disp': left_disp,
            'occ_mask': occ_mask,
        }

        if self.return_right_disp:
            right_disp_path = left_disp_path.replace('left', 'right')
            right_disp = Image.open(right_disp_path)
            right_disp = np.array(right_disp, dtype=np.float32)
            occ_mask_right = right_disp == 0
            right_disp = right_disp/100 
            sample['disp_right'] = right_disp
            sample['occ_mask_right'] = occ_mask_right

        if self.augmentations is not None:
            for t in self.augmentations:
                sample = t(sample)

        sample['index'] = idx
        sample['name'] = left_path

        return sample
    