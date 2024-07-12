# @Time    : 2023/8/26 14:23
# @Author  : zhangchenming
import os
import numpy as np
from PIL import Image
from pathlib import Path
from .utils.readpfm import readpfm
from .dataset_template import DatasetTemplate


def read_extrinsics(path, fid):
    with open(path) as f:
        line = f.readline()
        while line:
            if "Frame" in line:
                if fid == int(line[6:]):
                    line = f.readline()
                    left_extrinsics = np.array(line[2:])
                    line = f.readline()
                    right_extrinsics = np.array(line[2:])
                    return left_extrinsics, right_extrinsics
            line = f.readline()
        return None


class SceneFlowDataset(DatasetTemplate):
    def __init__(self, data_root_path, split_file, augmentations, return_right_disp, with_camera_data=False):
        super().__init__(data_root_path, split_file, augmentations)
        self.return_right_disp = return_right_disp
        self.with_camera_data = with_camera_data

        if self.with_camera_data:
            sample_list = []
            for each in self.data_list:
                full_paths = [os.path.join(self.root, x) for x in each[0:3]]
                left_img_path, right_img_path, disp_img_path = full_paths
                if "15mm" in left_img_path:
                    intrinsics = np.array([[450.0, 0.0, 479.5], [0.0, 450.0, 269.5], [0.0, 0.0, 1.0]]).astype(np.float32)
                else:
                    intrinsics = np.array([[1050.0, 0.0, 479.5], [0.0, 1050.0, 269.5], [0.0, 0.0, 1.0]]).astype(np.float32)
                camera_data_file = Path(disp_img_path.replace('disparity', 'camera_data')).parent.parent / 'camera_data.txt'
                extrinsics = read_extrinsics(camera_data_file, int(Path(left_img_path).stem))
                if extrinsics is None:
                    continue
                else:
                    left_extrinsics, right_extrinsics = extrinsics
                sample_list.append([*each[0:3], intrinsics, left_extrinsics, right_extrinsics])
            self.data_list = sample_list

    def __getitem__(self, idx):
        item = self.data_list[idx]
        full_paths = [os.path.join(self.root, x) for x in item[0:3]]
        left_img_path, right_img_path, disp_img_path = full_paths
        left_img = Image.open(left_img_path).convert('RGB')
        left_img = np.array(left_img, dtype=np.float32)
        right_img = Image.open(right_img_path).convert('RGB')
        right_img = np.array(right_img, dtype=np.float32)
        disp_img = readpfm(disp_img_path)[0].astype(np.float32)
        assert not np.isnan(disp_img).any(), 'disp_img has nan'
        sample = {
            'left': left_img,  # [H, W, 3]
            'right': right_img,  # [H, W, 3]
            'disp': disp_img,  # [H, W]
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

        left_extrinsics = np.fromstring(str(item[4]), dtype=float, sep=' ').astype(np.float32).reshape((4, 4))
        right_extrinsics = np.fromstring(str(item[5]), dtype=float, sep=' ').astype(np.float32).reshape((4, 4))
        sample['intrinsics'] = item[3]
        sample['left_extrinsics'] = left_extrinsics
        sample['right_extrinsics'] = right_extrinsics
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
