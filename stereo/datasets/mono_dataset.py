# @Time    : 2024/10/27 01:08
# @Author  : zhangchenming
import os
import random
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from .dataset_template import DatasetTemplate
from .mono import WarpDataset


class MonoDataset(DatasetTemplate):
    def __init__(self, data_root_path, split_file, augmentations):
        super().__init__(data_root_path, split_file, augmentations)
        self.warp = WarpDataset()

    def __getitem__(self, idx):
        item = self.data_list[idx]
        full_paths = [os.path.join(self.root, x) for x in item]
        left_path, depth_path = full_paths
        left_image = Image.open(left_path).convert('RGB')
        background_path, _ = random.choice(self.data_list)
        background_image = Image.open(background_path).convert('RGB')
        loaded_disparity = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        loaded_disparity = loaded_disparity.astype(np.float32) / 100
        inputs = {'left_image': left_image,
                  'background': background_image,
                  'loaded_disparity': loaded_disparity}
        inputs = self.warp.prepare_sizes(inputs)
        inputs['background'] = transfer_color(np.array(inputs['background']), np.array(inputs['left_image']))
        inputs['disparity'] = self.warp.process_disparity(inputs['loaded_disparity'], max_disparity_range=(50, 192))
        projection_disparity = inputs['disparity']
        right_image = self.warp.project_image(inputs['left_image'], projection_disparity, inputs['background'])
        sample = {
            'left': np.array(inputs['left_image'], dtype=np.float32),
            'right': right_image.astype(np.float32),
            'disp': inputs['disparity'].astype(np.float32),
            'occ_mask': np.zeros_like(inputs['disparity'], dtype=bool)
        }

        img = cv2.cvtColor(sample['left'], cv2.COLOR_RGB2BGR)
        lsc = cv2.ximgproc.createSuperpixelLSC(img, region_size=10, ratio=0.075)
        lsc.iterate(20)
        label = lsc.getLabels()
        sample['super_pixel_label'] = label.astype(np.int32)

        if self.augmentations is not None:
            for t in self.augmentations:
                sample = t(sample)

        sample['valid'] = sample['disp'] < 512
        sample['index'] = idx
        sample['name'] = left_path

        return sample


def transfer_color(target, source):
    target = target.astype(float) / 255
    source = source.astype(float) / 255

    target_means = target.mean(0).mean(0)
    target_stds = target.std(0).std(0)

    source_means = source.mean(0).mean(0)
    source_stds = source.std(0).std(0)

    target -= target_means
    target /= target_stds / source_stds
    target += source_means

    target = np.clip(target, 0, 1)
    target = (target * 255).astype(np.uint8)

    return target