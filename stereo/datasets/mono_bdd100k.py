# @Time    : 2024/10/27 01:08
# @Author  : zhangchenming
import os
import random
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from .dataset_template import DatasetTemplate


class BDD100K(DatasetTemplate):
    def __init__(self, data_root_path, split_file, augmentations):
        super().__init__(data_root_path, split_file, augmentations)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        full_paths = [os.path.join(self.root, x) for x in item]
        left_path, depth_path = full_paths
        left_image = Image.open(left_path).convert('RGB')
        background_path, _ = random.choice(self.data_list)
        background = Image.open(background_path).convert('RGB')

        loaded_disparity = np.array(Image.open(background_path))[0]

        print('fuck')
