import os
import argparse
import torch
import numpy as np

from PIL import Image

from stereo.config.lazy import LazyConfig, LazyCall
from stereo.config.instantiate import instantiate
from stereo.utils import common_utils
from stereo.datasets.utils import stereo_trans
from stereo.utils.disp_color import disp_to_color
from stereo.evaluation import metric_names, metric_funcs

from cfgs.common.constants import constants


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, required=True, help='specify the config for eval')
    parser.add_argument('--pretrained_model', type=str, default=None, required=True, help='pretrained_model')
    parser.add_argument('--left_img_path', type=str, default=None)
    parser.add_argument('--right_img_path', type=str, default=None)
    parser.add_argument('--disp_img_path', type=str, default=None)
    parser.add_argument('--savename', type=str, default=None)

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

    left_img_path = args.left_img_path
    right_img_path = args.right_img_path
    left_img = np.array(Image.open(left_img_path).convert('RGB'), dtype=np.float32)
    right_img = np.array(Image.open(right_img_path).convert('RGB'), dtype=np.float32)

    # disp_gt = np.array(Image.open(args.disp_img_path), dtype=np.float32)
    # full = True if 'full' in args.disp_img_path else False
    # scale = 128.0 if full else 256.0
    # disp_gt = disp_gt / scale

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
    # init_disp = model_pred['init_disp'].squeeze().cpu().numpy()

    pad_top, pad_right, pad_bottom, pad_left = sample['pad']
    h_start = pad_top
    h_end = None if pad_bottom == 0 else -pad_bottom
    w_start = pad_left
    w_end = None if pad_right == 0 else -pad_right
    disp_pred = disp_pred[slice(h_start, h_end), slice(w_start, w_end)]
    # init_disp = init_disp[slice(h_start, h_end), slice(w_start, w_end)]

    assert disp_pred.shape[0:2] == left_img.shape[0:2]

    # img = disp_pred.astype('uint8')
    # img = Image.fromarray(img)
    # img.save('uint8.png')

    print(disp_pred.max())
    # print('epe is {}'.format(metric_funcs['epe'](disp_pred, disp_gt)))
    img_color = disp_to_color(disp_pred, max_disp=192)
    img_color = img_color.astype('uint8')
    img_color = Image.fromarray(img_color)
    img_color.save(args.savename)

    # img_color = disp_to_color(init_disp, max_disp=192)
    # img_color = img_color.astype('uint8')
    # img_color = Image.fromarray(img_color)
    # img_color.save(args.savename[:-4] + '_init.png')


if __name__ == '__main__':
    main()
