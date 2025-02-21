# @Time    : 2023/10/17 16:18
# @Author  : zhangchenming
import os
import argparse
import datetime
import yaml
import torch
import torch.distributed as dist

from pathlib import Path
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

from stereo.config.instantiate import instantiate
from stereo.config.lazy import LazyConfig
from stereo.utils import common_utils
from stereo.solver.trainer import Trainer

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--dist_mode', action='store_true', default=False, help='torchrun ddp multi gpu')
    parser.add_argument('--cfg_file', type=str, default=None, required=True, help='specify the config for eval')
    parser.add_argument('--eval_data_cfg_file', type=str, default=None)
    parser.add_argument('--eval_batch_size', type=int, default=1)
    parser.add_argument('--pretrained_model', type=str, default=None, required=True, help='pretrained_model')
    parser.add_argument('--update', action='append', nargs=2, metavar=('KEY', 'VALUE'), help="Update a specific key in the configuration. Format: --update key value")

    args = parser.parse_args()
    cfg = LazyConfig.load(args.cfg_file)
    if not os.path.isfile(args.pretrained_model):
        raise FileNotFoundError
    cfg.runtime_params.pretrained_model = args.pretrained_model
    if args.eval_data_cfg_file:
        cfg.val_loader = LazyConfig.load(args.eval_data_cfg_file).val_loader
        cfg.val_loader.batch_size = args.eval_batch_size

    if args.update:
        for key, value in args.update:
            OmegaConf.update(cfg, key, yaml.safe_load(value), merge=True)

    args.run_mode = 'eval'
    return args, cfg


def main():
    args, cfg = parse_config()

    if args.dist_mode:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ["LOCAL_RANK"])
        global_rank = int(os.environ["RANK"])
    else:
        local_rank = 0
        global_rank = 0

    # env
    torch.cuda.set_device(local_rank)

    # savedir
    output_dir = str(Path(args.pretrained_model).parent)

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
    logfile_middle_name = Path(args.eval_data_cfg_file).stem if args.eval_data_cfg_file else 'default'
    log_file = os.path.join(output_dir, 'eval_{}_{}.log'.format(logfile_middle_name, shared_time))
    if global_rank == 0:
        open(log_file, "w").close()
    if args.dist_mode:
        dist.barrier()
    logger = common_utils.create_logger(log_file, rank=global_rank)

    # tensorboard
    tb_writer = SummaryWriter(log_dir=os.path.join(output_dir, 'eval_tensorboard')) if global_rank == 0 else None

    # log
    logger.info("Command line arguments: " + str(args))
    # with open(args.cfg_file, "r", encoding="utf-8") as f:
    #     content = f.read()
    # logger.info("Contents of args.config_file={}:\n{}".format(args.cfg_file, content))

    args.local_rank = local_rank
    args.global_rank = global_rank
    if 'trainer' in cfg:
        cfg.trainer.args = args
        cfg.trainer.cfg = cfg
        cfg.trainer.logger = logger
        cfg.trainer.tb_writer = tb_writer
        model_trainer = instantiate(cfg.trainer)
    else:
        model_trainer = Trainer(args, cfg, logger, tb_writer)

    model_trainer.evaluate(current_epoch=0)


if __name__ == '__main__':
    main()
