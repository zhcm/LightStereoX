# @Time    : 2023/8/26 14:23
# @Author  : zhangchenming
import os
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from .dataset_template import DatasetTemplate


class NERFStereoDataset(DatasetTemplate):
    def __init__(self, data_root_path, split_file, augmentations):
        super().__init__(data_root_path, split_file, augmentations)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        full_paths = [os.path.join(self.root, x) for x in item[0:3]]
        left_img_path, right_img_path, disp_img_path = full_paths

        left_path = Path(left_img_path)
        left_img_path = left_path.parent.parent.parent.joinpath('center').joinpath(left_path.name)
        assert left_img_path.exists(), str(left_img_path)
        relativepath = left_img_path.relative_to(Path(self.root))
        left_img_path = str(left_img_path)

        conf_img_path = left_path.parent.parent.parent.joinpath('AO').joinpath(left_path.stem + '.png')
        conf = cv2.imread(conf_img_path, -1) / 65536.0

        left_img = Image.open(left_img_path).convert('RGB')
        left_img = np.array(left_img, dtype=np.float32)
        right_img = Image.open(right_img_path).convert('RGB')
        right_img = np.array(right_img, dtype=np.float32)

        super_pixel_label = Path(self.root).parent.joinpath('SuperPixelLabel/NERFStereo', relativepath)
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

        # disp_img = np.array(Image.open(disp_img_path), dtype=np.float32)
        # disp_img = disp_img / 64.0
        disp_img = cv2.imread(disp_img_path, -1) / 64.
        disp_img = disp_img.astype(np.float32)

        disp_img = disp_img * (conf > 0.5)
        assert not np.isnan(disp_img).any(), 'disp_img has nan'
        occ_mask = np.zeros_like(disp_img, dtype=bool)
        sample = {
            'left': left_img,  # [H, W, 3]
            'right': right_img,  # [H, W, 3]
            'disp': disp_img,  # [H, W]
            'occ_mask': occ_mask,
            'super_pixel_label': super_pixel_label
        }

        if self.augmentations is not None:
            for t in self.augmentations:
                sample = t(sample)

        sample['valid'] = sample['disp'] < 512
        assert not sample['occ_mask'].any(), 'there is a True in Sceneflow occ mask'
        sample['index'] = idx
        sample['name'] = left_img_path
        return sample
