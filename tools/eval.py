# @Time    : 2023/10/17 16:18
# @Author  : zhangchenming
import sys
import os
import argparse
import datetime
import torch
import torch.distributed as dist
from easydict import EasyDict
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, './')
from stereo.utils import common_utils
from stereo.solver.trainer import Trainer

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--dist_mode', action='store_true', default=False, help='torchrun ddp multi gpu')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for eval')
    # dataloader
    parser.add_argument('--workers', type=int, default=0, help='number of workers for dataloader')
    parser.add_argument('--pin_memory', action='store_true', default=False, help='data loader pin memory')
    # parameters
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--eval_data_cfg_file', type=str, default=None)
    # interval
    parser.add_argument('--logger_iter_interval', type=int, default=1, help='')
    parser.add_argument('--eval_visualization', action='store_true', default=False, help='')

    args = parser.parse_args()
    yaml_config = common_utils.config_loader(args.cfg_file)
    cfgs = EasyDict(yaml_config)

    if args.eval_data_cfg_file:
        eval_data_yaml_config = common_utils.config_loader(args.eval_data_cfg_file)
        eval_data_cfgs = EasyDict(eval_data_yaml_config)
        cfgs.DATA_CONFIG = eval_data_cfgs.DATA_CONFIG
        cfgs.EVALUATOR = eval_data_cfgs.EVALUATOR

    args.run_mode = 'eval'
    return args, cfgs


def main():
    args, cfgs = parse_config()
    if args.dist_mode:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ["LOCAL_RANK"])
        global_rank = int(os.environ["RANK"])
    else:
        local_rank = 0
        global_rank = 0

    # env
    torch.cuda.set_device(local_rank)

    # log
    args.output_dir = str(Path(args.pretrained_model).parent.parent)
    log_file = os.path.join(args.output_dir, 'eval_%s.log' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=local_rank)
    tb_writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'eval_tensorboard')) if global_rank == 0 else None

    # log args and cfgs
    logger.info('LOGGING ARGS')
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    logger.info('LOGGING CFGS')
    common_utils.log_configs(cfgs, logger=logger)

    model_trainer = Trainer(args, cfgs, local_rank, global_rank, logger, tb_writer)
    model_trainer.evaluate(current_epoch=0)


if __name__ == '__main__':
    main()
