import os
import numpy as np
from PIL import Image
from .dataset_template import DatasetTemplate


class DrivingDataset(DatasetTemplate):
    def __init__(self, data_root_path, split_file, augmentations):
        super().__init__(data_root_path, split_file, augmentations)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        full_paths = [os.path.join(self.root, x) for x in item]
        left_img_path, right_img_path, disp_img_path = full_paths
        # image
        left_img = np.array(Image.open(left_img_path).convert('RGB'), dtype=np.float32)
        right_img = np.array(Image.open(right_img_path).convert('RGB'), dtype=np.float32)
        # disp
        disp_img = np.array(Image.open(disp_img_path), dtype=np.float32)
        # for validation, full resolution disp need to be divided by 128 instead of 256
        full = True if 'full' in disp_img_path else False
        scale = 128.0 if full else 256.0
        disp_img = disp_img / scale

        sample = {
            'left': left_img,
            'right': right_img,
            'disp': disp_img
        }

        if self.augmentations is not None:
            for t in self.augmentations:
                sample = t(sample)

        sample['index'] = idx
        sample['name'] = left_img_path
        return sample
