# @Time    : 2024/8/29 17:00
# @Author  : zhangchenming
import os
import numpy as np
import cv2
from pathlib import Path
from .dataset_template import DatasetTemplate


class ArgoverseDataset(DatasetTemplate):
    def __init__(self, data_root_path, split_file, augmentations):
        super().__init__(data_root_path, split_file, augmentations)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        full_paths = [os.path.join(self.root, x) for x in item]
        left_path, right_path, disp_path = full_paths

        left_img = cv2.cvtColor(cv2.imread(left_path), cv2.COLOR_BGR2RGB)

        right_img = cv2.cvtColor(cv2.imread(right_path), cv2.COLOR_BGR2RGB)

        disp_img = cv2.imread(disp_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        disp_img = np.float32(disp_img) / 256.0

        sample = {
            'left': left_img,
            'right': right_img,
            'disp': disp_img,
        }

        if self.augmentations is not None:
            for t in self.augmentations:
                sample = t(sample)

        sample['index'] = idx
        sample['name'] = left_path

        return sample
