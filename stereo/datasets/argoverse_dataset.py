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

        super_pixel_label = Path(self.root).parent.joinpath('SuperPixelLabel/Argoverse', item[0])
        super_pixel_label = str(super_pixel_label)[:-len('.png')] + "_lsc_lbl.png"
        if not os.path.exists(os.path.dirname(super_pixel_label)):
            os.makedirs(os.path.dirname(super_pixel_label), exist_ok=True)
        if not os.path.exists(super_pixel_label):
            img = cv2.cvtColor(left_img, cv2.COLOR_RGB2BGR)
            lsc = cv2.ximgproc.createSuperpixelLSC(img, region_size=10, ratio=0.075)
            lsc.iterate(20)
            label = lsc.getLabels()
            cv2.imwrite(super_pixel_label, label.astype(np.uint16))
        super_pixel_label = cv2.imread(super_pixel_label, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH).astype(np.int32)

        disp_img = cv2.imread(disp_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        disp_img = np.float32(disp_img) / 256.0
        occ_mask = np.zeros_like(disp_img, dtype=bool)

        sample = {
            'left': left_img,
            'right': right_img,
            'disp': disp_img,
            'occ_mask': occ_mask,
            'super_pixel_label': super_pixel_label
        }

        if self.augmentations is not None:
            for t in self.augmentations:
                sample = t(sample)

        sample['valid'] = sample['disp'] < 512
        sample['index'] = idx
        sample['name'] = left_path

        return sample
