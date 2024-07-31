import os
import numpy as np
from PIL import Image
from .dataset_template import DatasetTemplate


def disparity_read(filename):
    f_in = np.array(Image.open(filename))
    d_r = f_in[:, :, 0].astype('float64')
    d_g = f_in[:, :, 1].astype('float64')
    d_b = f_in[:, :, 2].astype('float64')

    disp = d_r * 4 + d_g / (2 ** 6) + d_b / (2 ** 14)
    return disp


class SintelDataset(DatasetTemplate):
    def __init__(self, data_root_path, split_file, augmentations):
        super().__init__(data_root_path, split_file, augmentations)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        full_paths = [os.path.join(self.root, x) for x in item]
        left_path, right_path, disp_path = full_paths

        left_img = Image.open(left_path).convert('RGB')
        left_img = np.array(left_img, dtype=np.float32)

        right_img = Image.open(right_path).convert('RGB')
        right_img = np.array(right_img, dtype=np.float32)

        disp_img = disparity_read(disp_path)

        occ_path = disp_path.replace('disparities', 'occlusions')
        occ = Image.open(occ_path)
        occ = np.array(occ, dtype=np.float32)

        sample = {
            'left': left_img,
            'right': right_img,
            'disp': disp_img,
            'occ_mask': occ
        }

        if self.augmentations is not None:
            for t in self.augmentations:
                sample = t(sample)

        sample['index'] = idx
        sample['name'] = left_path

        return sample
