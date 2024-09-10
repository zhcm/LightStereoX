# @Time    : 2023/10/8 15:01
# @Author  : zhangchenming
import argparse
import os
import numpy as np
import torch
import datetime
from pathlib import Path

from stereo.config.lazy import LazyConfig
from stereo.config.instantiate import instantiate
from stereo.utils import common_utils, flow_io


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, required=True, help='specify the config for eval')
    parser.add_argument('--pretrained_model', type=str, default=None, required=True, help='pretrained_model')

    args = parser.parse_args()
    cfg = LazyConfig.load(args.cfg_file)

    return args, cfg


def infer_and_save(dataloader, model, local_rank, cfg, logger, sprint_disp_dir):
    for i, data in enumerate(dataloader):
        for k, v in data.items():
            data[k] = v.to(local_rank) if torch.is_tensor(v) else v

        # left infer
        with torch.cuda.amp.autocast(enabled=cfg.runtime_params.mixed_precision):
            model_pred = model({'left': data['left'], 'right': data['right']})

        # pad
        pad_top, pad_right, pad_bottom, pad_left = data['pad'].squeeze().cpu().numpy()
        h_start = pad_top
        h_end = None if pad_bottom == 0 else -pad_bottom
        w_start = pad_left
        w_end = None if pad_right == 0 else -pad_right

        # orig size
        left_disp_pred = model_pred['disp_pred'].squeeze().cpu().numpy()
        left_disp_pred = left_disp_pred[slice(h_start, h_end), slice(w_start, w_end)]

        # save path
        left_split_name = data['name'][0].split('/')
        left_split_name.pop(0)
        left_name = '/'.join(left_split_name)
        left_name = left_name.replace('.png', '.dsp5').replace('frame', 'disp1')
        left_submit_path = os.path.join(sprint_disp_dir, left_name)
        os.makedirs(os.path.dirname(left_submit_path), exist_ok=True)

        # save
        flow_io.writeDispFile(left_disp_pred, left_submit_path)

        # right infer
        with torch.cuda.amp.autocast(enabled=cfg.runtime_params.mixed_precision):
            model_pred = model({'left': data['right_fl'], 'right': data['left_fl']})

        # orig size
        right_disp_pred = model_pred['disp_pred'].squeeze().cpu().numpy()
        right_disp_pred = right_disp_pred[slice(h_start, h_end), slice(w_start, w_end)]
        right_disp_pred = np.fliplr(right_disp_pred)

        # save path
        right_name = left_name.replace('left', 'right')
        right_submit_path = os.path.join(sprint_disp_dir, right_name)
        os.makedirs(os.path.dirname(right_submit_path), exist_ok=True)

        # save
        flow_io.writeDispFile(right_disp_pred, right_submit_path)

        message = 'Iter:{:>4d}/{}'.format(i, len(dataloader))
        logger.info(message)


@torch.no_grad()
def main():
    args, cfg = parse_config()
    local_rank = 0
    torch.cuda.set_device(local_rank)
    output_dir = str(Path(args.pretrained_model).parent)

    # logger
    log_file = os.path.join(output_dir, 'testspring_%s.log' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=local_rank)
    logger.info("Command line arguments: " + str(args))
    with open(args.cfg_file, "r", encoding="utf-8") as f:
        content = f.read()
    logger.info("Contents of args.config_file={}:\n{}".format(args.cfg_file, content))

    # model
    model = instantiate(cfg.model).to(local_rank)
    model.eval()

    # load pretrained model
    pretrained_model = args.pretrained_model
    if pretrained_model:
        print('Loading parameters from checkpoint %s' % pretrained_model)
        if not os.path.isfile(pretrained_model):
            raise FileNotFoundError
        common_utils.load_params_from_file(model, pretrained_model, device='cpu', logger=None, strict=True)

    spring_cfg = LazyConfig.load('cfgs/common/datasets/spring.py')
    spring_cfg.test_loader.is_dist = False
    _, spring_test_loader, _ = instantiate(spring_cfg.test_loader)

    sprint_disp_dir = os.path.join(output_dir, 'spring_disp')

    os.makedirs(sprint_disp_dir, exist_ok=True)

    infer_and_save(spring_test_loader, model, local_rank, cfg, logger, sprint_disp_dir)

    logger.info(os.path.abspath(sprint_disp_dir))


if __name__ == '__main__':
    main()
