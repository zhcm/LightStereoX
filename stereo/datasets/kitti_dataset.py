import os
import numpy as np

from PIL import Image
from .dataset_template import DatasetTemplate


class KittiDataset(DatasetTemplate):
    def __init__(self, data_root_path, split_file, augmentations, return_right_disp, use_noc):
        super().__init__(data_root_path, split_file, augmentations)
        self.return_right_disp = return_right_disp
        self.use_noc = use_noc

    def __getitem__(self, idx):
        item = self.data_list[idx]
        full_paths = [os.path.join(self.root, x) for x in item]
        left_img_path, right_img_path, disp_img_path = full_paths
        if self.use_noc:
            disp_img_path = disp_img_path.replace('disp_occ', 'disp_noc')
        # image
        left_img = np.array(Image.open(left_img_path).convert('RGB'), dtype=np.float32)
        right_img = np.array(Image.open(right_img_path).convert('RGB'), dtype=np.float32)
        # disp
        disp_img = np.array(Image.open(disp_img_path), dtype=np.float32) / 256.0

        sample = {
            'left': left_img,
            'right': right_img,
            'disp': disp_img,
        }
        if self.return_right_disp:
            disp_img_right_path = disp_img_path.replace('c_0', 'c_1')
            disp_img_right = np.array(Image.open(disp_img_right_path), dtype=np.float32) / 256.0
            sample['disp_right'] = disp_img_right

        for t in self.augmentations:
            sample = t(sample)

        sample['index'] = idx
        sample['name'] = left_img_path

        return sample


class KittiTestDataset(DatasetTemplate):
    def __init__(self, data_root_path, split_file, augmentations):
        super().__init__(data_root_path, split_file, augmentations)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        full_paths = [os.path.join(self.root, x) for x in item]
        left_img_path, right_img_path = full_paths[:2]
        left_img = np.array(Image.open(left_img_path).convert('RGB'), dtype=np.float32)
        right_img = np.array(Image.open(right_img_path).convert('RGB'), dtype=np.float32)
        sample = {
            'left': left_img,
            'right': right_img,
            'name': left_img_path.split('/')[-1],
        }

        for t in self.augmentations:
            sample = t(sample)

        return sample
