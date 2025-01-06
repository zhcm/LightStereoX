# @Time    : 2024/11/1 07:52
# @Author  : zhangchenming
import os
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from .dataset_template import DatasetTemplate


class SpeedBump(DatasetTemplate):
    def __init__(self, data_root_path, split_file, augmentations):
        super().__init__(data_root_path, split_file, augmentations)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        left_path, right_path, disp_path, seg_path, height, baseline, focallength, height_path = item
        height_map = np.load(height_path).astype(np.float32)

        left_img = Image.open(left_path).convert('RGB')
        left_img = np.array(left_img, dtype=np.float32)

        right_img = Image.open(right_path).convert('RGB')
        right_img = np.array(right_img, dtype=np.float32)

        disp = np.array(Image.open(disp_path), dtype=np.float32)
        seg = np.array(Image.open(seg_path), dtype=np.float32)
        bump_mask = seg == 230
        height = float(height) / 100
        assert bump_mask.any()

        sample = {
            'left': left_img,
            'right': right_img,
            'disp': disp,
            'bump_mask': bump_mask,
            'height_map': height_map,
            # 'super_pixel_label': super_pixel_label
        }

        if self.augmentations is not None:
            for t in self.augmentations:
                sample = t(sample)

        sample['dilated_bump_mask'] = np.zeros_like(sample['bump_mask'])
        if sample['bump_mask'].any():
            true_coords = np.argwhere(sample['bump_mask'])
            min_y = true_coords[:, 0].min()
            max_y = true_coords[:, 0].max()
            min_x = true_coords[:, 1].min()
            max_x = true_coords[:, 1].max()
            new_min_y = max(0, int(min_y - (max_y - min_y) / 5))
            new_max_y = min(sample['bump_mask'].shape[0], int(max_y + (max_y - min_y) / 5))
            sample['dilated_bump_mask'][new_min_y:new_max_y, min_x:max_x] = True

        sample['bump_height_map'] = sample['height_map'].copy()
        sample['bump_height_map'][~sample['bump_mask']] = 0
        sample['occ_mask'] = ~sample['bump_mask']

        sample['valid'] = sample['disp'] < 512
        sample['height'] = height
        sample['baseline'] = float(baseline)
        sample['focallength'] = float(focallength)

        sample['index'] = idx

        return sample
