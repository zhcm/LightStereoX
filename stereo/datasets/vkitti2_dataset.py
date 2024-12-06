import os
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from .dataset_template import DatasetTemplate


class VirtualKitti2Dataset(DatasetTemplate):
    def __init__(self, data_root_path, split_file, augmentations, return_right_disp):
        super().__init__(data_root_path, split_file, augmentations)
        self.return_right_disp = return_right_disp

    def __getitem__(self, idx):
        item = self.data_list[idx]
        full_paths = [os.path.join(self.root, x) for x in item[0:4]]
        left_img_path, right_img_path, disp_img_path, disp_img_right_path = full_paths
        left_img = Image.open(left_img_path).convert('RGB')
        left_img = np.array(left_img, dtype=np.float32)
        right_img = Image.open(right_img_path).convert('RGB')
        right_img = np.array(right_img, dtype=np.float32)
        disp_img = get_disp(disp_img_path).astype(np.float32)
        assert not np.isnan(disp_img).any(), 'disp_img has nan'
        occ_mask = np.zeros_like(disp_img, dtype=bool)
        sample = {
            'left': left_img,  # [H, W, 3]
            'right': right_img,  # [H, W, 3]
            'disp': disp_img,  # [H, W]
            'occ_mask': occ_mask
        }
        if self.return_right_disp:
            disp_img_right = get_disp(disp_img_right_path).astype(np.float32)
            sample['disp_right'] = disp_img_right
            assert not np.isnan(disp_img_right).any(), 'disp_img_right has nan'

        if self.augmentations is not None:
            for t in self.augmentations:
                sample = t(sample)

        sample['index'] = idx
        sample['name'] = left_img_path
        return sample


def get_disp(file_path, checkinvalid=True):
    if '.png' in file_path:
        depth = cv2.imread(file_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
        invalid = depth >= 65535
        num_invalid = depth[invalid].shape[0]
        depth = depth / 100.0
    else:
        raise TypeError('only support png and npy format, invalid type found: {}'.format(file_path))

    f = 725.0087
    b = 0.532725  # meter

    disp = b * f / (depth + 1e-5)
    if checkinvalid:
        disp[invalid] = 0
    return disp
