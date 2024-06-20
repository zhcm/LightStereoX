# @Time    : 2023/8/28 22:18
# @Author  : zhangchenming
import sys
import os
import argparse
import datetime
import tqdm
from easydict import EasyDict

import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, './')
from stereo.utils import common_utils
from trainer import Trainer

from stereo.config.lazy import LazyConfig


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    # mode
    parser.add_argument('--dist_mode', action='store_true', default=False, help='torchrun ddp multi gpu')
    parser.add_argument('--cfg_file', type=str, default=None, required=True, help='specify the config for training')
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')
    # save path
    parser.add_argument('--save_root_dir', type=str, default='./output', help='save root dir for this experiment')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    # dataloader
    parser.add_argument('--workers', type=int, default=0, help='number of workers for dataloader')
    parser.add_argument('--pin_memory', action='store_true', default=False, help='data loader pin memory')

    args = parser.parse_args()
    yaml_config = common_utils.config_loader(args.cfg_file)
    cfgs = EasyDict(yaml_config)
    args.run_mode = 'train'
    return args, cfgs


def main():
    cfg = LazyConfig.load('cfgs/lightstereo/lightstereo_s_sceneflow.py')
    args, cfgs = parse_config()
    if args.dist_mode:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ["LOCAL_RANK"])  # The local rank.
        global_rank = int(os.environ["RANK"])  # The global rank.
    else:
        local_rank = 0
        global_rank = 0

    # env
    torch.cuda.set_device(local_rank)
    if cfg.train_params.fix_random_seed:
        seed = 0 if not args.dist_mode else dist.get_rank()
        common_utils.set_random_seed(seed=seed)

    # savedir
    output_dir = str(os.path.join(cfg.train_params.save_root_dir, args.extra_tag))
    if os.path.exists(output_dir) and args.extra_tag != 'debug' and cfg.train_params.resume_from_ckpt == -1:
        raise Exception('There is already an exp with this name')
    if args.dist_mode:
        dist.barrier()
    args.ckpt_dir = os.path.join(output_dir, 'ckpt')
    if global_rank == 0:
        os.makedirs(args.ckpt_dir, exist_ok=True)
        common_utils.backup_source_code(os.path.join(output_dir, 'code'))
        os.system('cp %s %s' % (args.cfg_file, output_dir))
    if args.dist_mode:
        dist.barrier()

    # logger
    if global_rank == 0:
        now = datetime.datetime.now()
        timestamp = now.timestamp()
    else:
        timestamp = 0.0
    timestamp_tensor = torch.tensor([timestamp], dtype=torch.float64).cuda()
    if args.dist_mode:
        dist.broadcast(timestamp_tensor, src=0)
    shared_time = datetime.datetime.fromtimestamp(timestamp_tensor.item()).strftime('%Y%m%d-%H%M%S')
    log_file = os.path.join(output_dir, 'train_{}.log'.format(shared_time))
    if global_rank == 0:
        open(log_file, "w").close()
    if args.dist_mode:
        dist.barrier()
    logger = common_utils.create_logger(log_file, rank=global_rank)
    # tensorboard
    tb_writer = SummaryWriter(log_dir=os.path.join(output_dir, 'tensorboard')) if global_rank == 0 else None

    # trainer
    args.local_rank = local_rank
    args.global_rank = global_rank
    model_trainer = Trainer(args, cfgs, local_rank, global_rank, logger, tb_writer, cfg)

    tbar = tqdm.trange(model_trainer.last_epoch + 1, model_trainer.total_epochs,
                       desc='epochs', dynamic_ncols=True, disable=(local_rank != 0),
                       bar_format='{l_bar}{bar}{r_bar}\n')
    # train loop
    for current_epoch in tbar:
        model_trainer.train(current_epoch, tbar)
        model_trainer.save_ckpt(current_epoch)
        if current_epoch % cfgs.TRAINER.EVAL_INTERVAL == 0 or current_epoch == model_trainer.total_epochs - 1:
            model_trainer.evaluate(current_epoch)


if __name__ == '__main__':
    main()
