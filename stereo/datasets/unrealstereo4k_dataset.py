import os
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from .dataset_template import DatasetTemplate


class UnrealStereo4KDataset(DatasetTemplate):
    def __init__(self, data_root_path, split_file, augmentations, return_right_disp):
        super().__init__(data_root_path, split_file, augmentations)
        self.return_right_disp = return_right_disp

    def __getitem__(self, idx):
        item = self.data_list[idx]
        full_paths = [os.path.join(self.root, x) for x in item]
        left_path, right_path, disp_left_path = full_paths

        left_img = Image.open(left_path).convert('RGB')
        left_img = np.array(left_img, dtype=np.float32)

        right_img = Image.open(right_path).convert('RGB')
        right_img = np.array(right_img, dtype=np.float32)

        super_pixel_label = Path(self.root).parent.joinpath('SuperPixelLabel/UnrealStereo4K', item[0])
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

        disp_left = np.load(disp_left_path, mmap_mode='c')
        occ_mask = np.zeros_like(disp_left, dtype=bool)

        sample = {
            'left': left_img,
            'right': right_img,
            'disp': disp_left,
            'occ_mask': occ_mask,
            'super_pixel_label': super_pixel_label
        }

        if self.return_right_disp:
            disp_right_path = disp_left_path.replace('Image0', 'Image1')
            disp_right = np.load(disp_right_path, mmap_mode='c')
            sample['disp_right'] = disp_right

        if self.augmentations is not None:
            for t in self.augmentations:
                sample = t(sample)

        sample['valid'] = sample['disp'] < 512
        sample['index'] = idx
        sample['name'] = left_path

        return sample
