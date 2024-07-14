# @Time    : 2023/10/8 15:01
# @Author  : zhangchenming
import argparse
import os
import torch
import datetime
from PIL import Image
from pathlib import Path

from stereo.config.lazy import LazyConfig
from stereo.config.instantiate import instantiate
from stereo.utils import common_utils


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, required=True, help='specify the config for eval')
    parser.add_argument('--pretrained_model', type=str, default=None, required=True, help='pretrained_model')

    args = parser.parse_args()
    cfg = LazyConfig.load(args.cfg_file)

    return args, cfg


def infer_and_save(dataloader, model, local_rank, cfg, logger, result_dir):
    for i, data in enumerate(dataloader):
        for k, v in data.items():
            data[k] = v.to(local_rank) if torch.is_tensor(v) else v

        # infer
        with torch.cuda.amp.autocast(enabled=cfg.runtime_params.mixed_precision):
            model_pred = model(data)

        disp_pred = model_pred['disp_pred'].squeeze().cpu().numpy()
        pad_top, pad_right, pad_bottom, pad_left = data['pad'].squeeze().cpu().numpy()
        h_start = pad_top
        h_end = None if pad_bottom == 0 else -pad_bottom
        w_start = pad_left
        w_end = None if pad_right == 0 else -pad_right
        disp_pred = disp_pred[slice(h_start, h_end), slice(w_start, w_end)]

        # save to file
        img = (disp_pred * 256).astype('uint16')
        img = Image.fromarray(img)
        name = data['name'][0]
        img.save(os.path.join(result_dir, name))
        message = 'Iter:{:>4d}/{}'.format(i, len(dataloader))
        logger.info(message)


@torch.no_grad()
def main():
    args, cfg = parse_config()
    local_rank = 0
    torch.cuda.set_device(local_rank)
    output_dir = str(Path(args.pretrained_model).parent)

    # logger
    log_file = os.path.join(output_dir, 'testkitti_%s.log' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
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

    # kitti15
    kitti15_cfg = LazyConfig.load('cfgs/common/datasets/kitti15.py')
    kitti15_cfg.test_loader.is_dist = False
    _, kitti15_test_loader, _ = instantiate(kitti15_cfg.test_loader)
    kitti15_result_dir = os.path.join(output_dir, 'disp_0')
    if not os.path.exists(kitti15_result_dir):
        os.makedirs(kitti15_result_dir)
    infer_and_save(kitti15_test_loader, model, local_rank, cfg, logger, kitti15_result_dir)

    # kitti12
    kitti12_cfg = LazyConfig.load('cfgs/common/datasets/kitti12.py')
    kitti12_cfg.test_loader.is_dist = False
    _, kitti12_test_loader, _ = instantiate(kitti12_cfg.test_loader)
    kitti12_result_dir = os.path.join(output_dir, 'disp_0_12')
    if not os.path.exists(kitti12_result_dir):
        os.makedirs(kitti12_result_dir)
    infer_and_save(kitti12_test_loader, model, local_rank, cfg, logger, kitti12_result_dir)

    logger.info(os.path.abspath(kitti15_result_dir))
    logger.info(os.path.abspath(kitti12_result_dir))


if __name__ == '__main__':
    main()
