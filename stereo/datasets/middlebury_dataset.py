import os
import numpy as np
from PIL import Image
from pathlib import Path
from .utils.readpfm import readpfm
from .dataset_template import DatasetTemplate


class MiddleburyDataset(DatasetTemplate):
    def __init__(self, data_root_path, split_file, augmentations):
        super().__init__(data_root_path, split_file, augmentations)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        full_paths = [os.path.join(self.root, x) for x in item]
        left_img_path, right_img_path, disp_img_path = full_paths

        left_img = Image.open(left_img_path).convert('RGB')
        left_img = np.array(left_img, dtype=np.float32)
        right_img = Image.open(right_img_path).convert('RGB')
        right_img = np.array(right_img, dtype=np.float32)
        disp_img = readpfm(disp_img_path)[0].astype(np.float32)
        disp_img[disp_img == np.inf] = 0

        occ_mask_path = left_img_path.replace('im0.png', 'mask0nocc.png')
        occ_mask = Image.open(occ_mask_path).convert('L')
        occ_mask = np.array(occ_mask, dtype=np.float32)
        occ_mask = occ_mask != 255.0

        sample = {
            'left': left_img,  # [H, W, 3]
            'right': right_img,  # [H, W, 3]
            'disp': disp_img,  # [H, W]
            'occ_mask': occ_mask  # [H, W]
        }

        if self.augmentations is not None:
            for t in self.augmentations:
                sample = t(sample)

        calib_path = Path(left_img_path).parent.joinpath('calib.txt')
        with open(calib_path) as f:
            line = f.readline()
            while line:
                k, v = line.split('=')
                if k == 'cam0':
                    rows = v.strip('[]\n').split(';')
                    matrix = [list(map(float, row.split())) for row in rows]
                    intrinsics = np.array(matrix).astype(np.float32)
                    sample['intrinsics'] = intrinsics
                if k == 'baseline':
                    sample['baseline'] = np.array(float(v)).astype(np.float32)

                line = f.readline()

        sample['index'] = idx
        sample['name'] = left_img_path
        return sample
