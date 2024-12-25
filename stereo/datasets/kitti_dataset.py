import os
import numpy as np
import cv2

from pathlib import Path
from PIL import Image
from .dataset_template import DatasetTemplate


def read_calib_file(path):
    # taken from https://github.com/hunse/kitti
    float_chars = set("0123456789.e+- ")
    data = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            value = value.strip()
            data[key] = value
            if float_chars.issuperset(value):
                # try to cast to float array
                try:
                    data[key] = np.array(value.split(' '), dtype='float32')
                except ValueError:
                    # casting error: data[key] already eq. value, so pass
                    pass
    return data


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

        super_pixel_label = Path(self.root).parent.joinpath('SuperPixelLabel/Kitti', item[0])
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

        # disp
        disp_img = np.array(Image.open(disp_img_path), dtype=np.float32) / 256.0
        occ_mask = np.zeros_like(disp_img, dtype=bool)

        sample = {
            'left': left_img,
            'right': right_img,
            'disp': disp_img,
            'occ_mask': occ_mask,
            'super_pixel_label': super_pixel_label
        }
        if self.return_right_disp:
            disp_img_right_path = disp_img_path.replace('c_0', 'c_1')
            disp_img_right = np.array(Image.open(disp_img_right_path), dtype=np.float32) / 256.0
            sample['disp_right'] = disp_img_right

        if self.augmentations is not None:
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
