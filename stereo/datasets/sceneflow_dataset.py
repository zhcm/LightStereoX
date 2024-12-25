# @Time    : 2023/8/26 14:23
# @Author  : zhangchenming
import os
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from .utils.readpfm import readpfm
from .dataset_template import DatasetTemplate


class SceneFlowDataset(DatasetTemplate):
    def __init__(self, data_root_path, split_file, augmentations, return_right_disp):
        super().__init__(data_root_path, split_file, augmentations)
        self.return_right_disp = return_right_disp

    def __getitem__(self, idx):
        item = self.data_list[idx]
        full_paths = [os.path.join(self.root, x) for x in item[0:3]]
        left_img_path, right_img_path, disp_img_path = full_paths
        left_img = Image.open(left_img_path).convert('RGB')
        left_img = np.array(left_img, dtype=np.float32)
        right_img = Image.open(right_img_path).convert('RGB')
        right_img = np.array(right_img, dtype=np.float32)

        super_pixel_label = Path(self.root).parent.joinpath('SuperPixelLabel/SceneFlow', item[0])
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

        disp_img = readpfm(disp_img_path)[0].astype(np.float32)
        assert not np.isnan(disp_img).any(), 'disp_img has nan'
        occ_mask = np.zeros_like(disp_img, dtype=bool)
        sample = {
            'left': left_img,  # [H, W, 3]
            'right': right_img,  # [H, W, 3]
            'disp': disp_img,  # [H, W]
            'occ_mask': occ_mask,
            'super_pixel_label': super_pixel_label
        }
        if self.return_right_disp:
            disp_img_right_path = disp_img_path.replace('left', 'right')
            disp_img_right = readpfm(disp_img_right_path)[0].astype(np.float32)
            disp_img_right = disp_img_right.astype(np.float32)
            sample['disp_right'] = disp_img_right
            assert not np.isnan(disp_img_right).any(), 'disp_img_right has nan'

        if self.augmentations is not None:
            for t in self.augmentations:
                sample = t(sample)

        sample['valid'] = sample['disp'] < 512
        assert not sample['occ_mask'].any(), 'there is a True in Sceneflow occ mask'
        sample['index'] = idx
        sample['name'] = left_img_path
        return sample


class FlyingThings3DSubsetDataset(DatasetTemplate):
    def __init__(self, data_root_path, split_file, augmentations, return_occ_mask, zeroing_occ):
        super().__init__(data_root_path, split_file, augmentations)
        self.return_occ_mask = return_occ_mask
        self.zeroing_occ = zeroing_occ

    def __getitem__(self, idx):
        item = self.data_list[idx]
        full_paths = [os.path.join(self.root, x) for x in item[0:6]]
        left_img_path, right_img_path, disp_img_path, disp_img_right_path, occ_path, occ_right_path = full_paths

        left_img = Image.open(left_img_path).convert('RGB')
        left_img = np.array(left_img, dtype=np.float32)
        right_img = Image.open(right_img_path).convert('RGB')
        right_img = np.array(right_img, dtype=np.float32)

        disp_img = readpfm(disp_img_path)[0].astype(np.float32)
        disp_img = np.nan_to_num(disp_img, nan=0.0)
        disp_img_right = readpfm(disp_img_right_path)[0].astype(np.float32)
        disp_img_right = np.nan_to_num(disp_img_right, nan=0.0)

        sample = {
            'left': left_img,  # [H, W, 3]
            'right': right_img,  # [H, W, 3]
            'disp': disp_img,  # [H, W]
            'disp_right': disp_img_right,  # [H, W]
        }

        if self.return_occ_mask:
            occ = np.array(Image.open(occ_path)).astype(np.bool_)
            occ_right = np.array(Image.open(occ_right_path)).astype(np.bool_)
            sample.update({
                'occ_mask': occ,  # [H, W]
                'occ_mask_right': occ_right  # [H, W]
            })

        if self.zeroing_occ:
            sample = self.make_occ_disp_zero(sample)

        for t in self.augmentations:
            sample = t(sample)

        sample['index'] = idx
        sample['name'] = left_img_path

        return sample

    def make_occ_disp_zero(self, input_data):
        w = input_data['disp'].shape[-1]
        input_data['disp'][input_data['disp'] > w] = 0
        input_data['disp'][input_data['disp'] < 0] = 0

        # manually compute occ area (this is necessary after cropping)
        occ_mask = self.compute_left_occ_region(w, input_data['disp'])
        input_data['occ_mask'][occ_mask] = True
        input_data['occ_mask'] = np.ascontiguousarray(input_data['occ_mask'])

        # manually compute occ area for right image
        try:
            occ_mask = self.compute_right_occ_region(w, input_data['disp_right'])
            input_data['occ_mask_right'][occ_mask] = True
            input_data['occ_mask_right'] = np.ascontiguousarray(input_data['occ_mask_right'])
        except KeyError:
            # print('No disp mask right, check if dataset is KITTI')
            input_data['occ_mask_right'] = np.zeros_like(occ_mask).astype(np.bool_)
        input_data.pop('disp_right', None)  # remove disp right after finish

        # set occlusion area to 0
        input_data['disp'][input_data['occ_mask']] = 0
        input_data['disp'] = np.ascontiguousarray(input_data['disp'], dtype=np.float32)

        return input_data

    @staticmethod
    def compute_left_occ_region(w, disp):
        """
        Compute occluded region on the left image border
        :param w: image width
        :param disp: left disparity
        :return: occ mask
        """
        coord = np.linspace(0, w - 1, w)[None,]  # [1, w]
        shifted_coord = coord - disp  # 通过视差找到右图对应的点
        occ_mask = shifted_coord < 0  # 判断是否在图片之外
        return occ_mask

    @staticmethod
    def compute_right_occ_region(w, disp):
        """
        Compute occluded region on the right image border
        :param w: image width
        :param disp: right disparity
        :return: occ mask
        """
        coord = np.linspace(0, w - 1, w)[None,]  # [1, w]
        shifted_coord = coord + disp  # 通过视差找到左图对应的点
        occ_mask = shifted_coord > w  # 判断是否在图片之外
        return occ_mask
