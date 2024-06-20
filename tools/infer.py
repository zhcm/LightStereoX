import glob
import sys
import os
import argparse
import datetime
import thop
import torch
import numpy as np
from PIL import Image
from easydict import EasyDict
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, './')
from stereo.utils import common_utils
from stereo.datasets import build_dataloader
from stereo.modeling import build_network
from stereo.datasets.dataset_template import build_transform_by_cfg
from stereo.utils.common_utils import load_params_from_file

from stereo.utils.disp_color import disp_to_color


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    # parameters
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--data_cfg_file', type=str, default=None)

    args = parser.parse_args()
    args.output_dir = str(Path(args.pretrained_model).parent.parent)
    yaml_files = glob.glob(os.path.join(args.output_dir, '*.yaml'), recursive=False)
    args.cfg_file = yaml_files[0]
    yaml_config = common_utils.config_loader(args.cfg_file)
    cfgs = EasyDict(yaml_config)
    return args, cfgs


@torch.no_grad()
def main():
    args, cfgs = parse_config()

    # env
    local_rank = 0
    torch.cuda.set_device(local_rank)
    common_utils.set_random_seed(seed=0)

    model = build_network(model_cfg=cfgs.MODEL).cuda()

    # load pretrained model
    if args.pretrained_model is not None:
        if not os.path.isfile(args.pretrained_model):
            raise FileNotFoundError
        print('Loading parameters from checkpoint %s' % args.pretrained_model)
        load_params_from_file(model, args.pretrained_model, device='cuda:%d' % local_rank,
                              dist_mode=False, logger=None, strict=False)

    data_cfgs = EasyDict(common_utils.config_loader(args.data_cfg_file))

    transform_config = data_cfgs.DATA_CONFIG.DATA_TRANSFORM.TESTING
    transform = build_transform_by_cfg(transform_config)

    root_dir = '/mnt/nas/algorithm/chenming.zhang/misc'
    for each in os.listdir(os.path.join(root_dir, 'left')):
        left_img_path = os.path.join(root_dir, 'left', each)
        right_img_path = os.path.join(root_dir, 'right', each)
        left_img = np.array(Image.open(left_img_path).convert('RGB'), dtype=np.float32)
        right_img = np.array(Image.open(right_img_path).convert('RGB'), dtype=np.float32)
        sample = {
            'left': left_img,
            'right': right_img,
        }
        sample = transform(sample)
        sample['left'] = sample['left'].unsqueeze(0)
        sample['right'] = sample['right'].unsqueeze(0)

        model.eval()
        for k, v in sample.items():
            sample[k] = v.to(local_rank) if torch.is_tensor(v) else v

        model_pred = model(sample)
        disp_pred = model_pred['disp_pred'].squeeze(1)
        pad_top, pad_right, _, _ = sample['pad']
        if pad_right == 0:
            disp_pred = disp_pred[:, pad_top:, :]
        else:
            disp_pred = disp_pred[:, pad_top:, :-pad_right]

        print(disp_pred.max())
        img_color = disp_to_color(disp_pred.squeeze(0).cpu().numpy(), max_disp=192)
        img_color = img_color.astype('uint8')
        img_color = Image.fromarray(img_color)
        img_color.save(os.path.join('/mnt/nas/algorithm/chenming.zhang/misc/result_color', os.path.splitext(each)[0] + '.png'))

        img = disp_pred.squeeze(0).cpu().numpy()
        img = (img * 256).astype('uint16')
        img = Image.fromarray(img)
        img.save(os.path.join('/mnt/nas/algorithm/chenming.zhang/misc/result', os.path.splitext(each)[0]+'.png'))


if __name__ == '__main__':
    main()