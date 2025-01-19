# @Time    : 2025/1/17 13:29
# @Author  : zhangchenming
import cv2
import glob
import os
import argparse
import torch
import numpy as np
import json
import open3d as o3d

from PIL import Image
from pathlib import Path

from stereo.config.lazy import LazyConfig, LazyCall
from stereo.config.instantiate import instantiate
from stereo.utils import common_utils
from stereo.datasets.utils import stereo_trans

from cfgs.common.constants import constants

root_dir = '/mnt/nas/algorithm/chenming.zhang/misc/demo_241226'


def split_img():
    file_list = glob.glob(os.path.join(root_dir, 'projector_image_undistorted_stereo/*.png'))
    for img_path in file_list:
        filename = Path(img_path).name
        img = cv2.imread(img_path)
        left_img = img[0:480, :, :]
        right_img = img[480:, :, :]
        cv2.imwrite(os.path.join(root_dir, 'left', filename), left_img)
        cv2.imwrite(os.path.join(root_dir, 'right', filename), right_img)


def readply():
    point_cloud = o3d.io.read_point_cloud(os.path.join(root_dir, 'demo_model.ply'))
    print("点云数据：", point_cloud)


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, required=True, help='specify the config for eval')
    parser.add_argument('--pretrained_model', type=str, default=None, required=True, help='pretrained_model')

    args = parser.parse_args()
    cfg = LazyConfig.load(args.cfg_file)

    return args, cfg


@torch.no_grad()
def main():
    args, cfg = parse_config()

    local_rank = 0
    torch.cuda.set_device(local_rank)

    model = instantiate(cfg.model).to(local_rank)
    model.eval()

    # load pretrained model
    pretrained_model = args.pretrained_model
    if pretrained_model:
        print('Loading parameters from checkpoint %s' % pretrained_model)
        if not os.path.isfile(pretrained_model):
            raise FileNotFoundError
        common_utils.load_params_from_file(model, pretrained_model, device='cpu', logger=None, strict=True)

    # 数据预处理
    pre_process = [
        LazyCall(stereo_trans.DivisiblePad)(divisor=32, mode='tr'),
        LazyCall(stereo_trans.NormalizeImage)(mean=constants.rgb_mean, std=constants.rgb_std)
    ]
    preprocessing = instantiate(pre_process)

    file_list = glob.glob(os.path.join(root_dir, 'left/*.png'))
    for left_img_path in file_list:
        stem = Path(left_img_path).stem
        pose_path = os.path.join(root_dir, 'pose', stem + '.json')

        right_img_path = left_img_path.replace('left', 'right')
        left_img = np.array(Image.open(left_img_path).convert('RGB'), dtype=np.float32)
        right_img = np.array(Image.open(right_img_path).convert('RGB'), dtype=np.float32)

        sample = {
            'left': left_img,
            'right': right_img,
        }
        for t in preprocessing:
            sample = t(sample)

        sample['left'] = torch.from_numpy(sample['left']).unsqueeze(0).to(local_rank)
        sample['right'] = torch.from_numpy(sample['right']).unsqueeze(0).to(local_rank)

        model_pred = model(sample)
        disp_pred = model_pred['disp_pred'].squeeze().cpu().numpy()

        pad_top, pad_right, pad_bottom, pad_left = sample['pad']
        h_start = pad_top
        h_end = None if pad_bottom == 0 else -pad_bottom
        w_start = pad_left
        w_end = None if pad_right == 0 else -pad_right
        disp_pred = disp_pred[slice(h_start, h_end), slice(w_start, w_end)]
        assert disp_pred.shape[0:2] == left_img.shape[0:2]

        if not os.path.exists(pose_path):
            f = 184.75209045410156
            B = 0.06400000303983688
        else:
            with open(pose_path, 'r') as f:
                posedata = json.load(f)
            f = posedata['rectified_stacked_stereo_intrinsic'][0]  # 焦距（像素）
            B = abs(posedata['translation'][1])  # 基线（米）

        depth_map = np.zeros_like(disp_pred, dtype=np.float32)
        valid_pixels = disp_pred > 0
        depth_map[valid_pixels] = (f * B) / disp_pred[valid_pixels]

        np.save(os.path.join(root_dir, 'disp', stem + '.npy'), disp_pred)
        np.save(os.path.join(root_dir, 'depth', stem + '.npy'), depth_map)


if __name__ == '__main__':
    # main()
    readply()
