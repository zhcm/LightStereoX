# @Time    : 2023/8/28 22:18
# @Author  : zhangchenming
import os
import argparse
import datetime
import tqdm
import shutil
import torch
import torch.distributed as dist

from torch.utils.tensorboard import SummaryWriter
from stereo.utils import common_utils
from stereo.solver.trainer import Trainer
from stereo.config.lazy import LazyConfig
from stereo.config.instantiate import instantiate

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--dist_mode', action='store_true', default=False, help='torchrun ddp multi gpu')
    parser.add_argument('--cfg_file', type=str, default=None, required=True, help='specify the config for training')
    parser.add_argument('--extra_tag', type=str, default='debug', help='extra tag for this experiment')
    parser.add_argument('--cover_old_exp', action='store_true', default=False)

    args = parser.parse_args()
    cfg = LazyConfig.load(args.cfg_file)

    args.run_mode = 'train'
    return args, cfg


def main():
    args, cfg = parse_config()

    # dist.get_rank() == int(os.environ["RANK"]) == global_rank
    if args.dist_mode:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ["LOCAL_RANK"])
        global_rank = int(os.environ["RANK"])
    else:
        local_rank = 0
        global_rank = 0

    # env
    torch.cuda.set_device(local_rank)
    if cfg.runtime_params.fix_random_seed:
        seed = 0 if not args.dist_mode else dist.get_rank()
        common_utils.set_random_seed(seed=seed, deterministic=False)

    # savedir
    output_dir = str(os.path.join(cfg.runtime_params.save_root_dir, args.extra_tag))
    if os.path.exists(output_dir) and args.cover_old_exp and global_rank == 0:
        shutil.rmtree(output_dir)
    if args.dist_mode:
        dist.barrier()
    if os.path.exists(output_dir) and args.extra_tag != 'debug' and cfg.runtime_params.resume_from_ckpt == -1:
        raise Exception('There is already an exp with this name')
    if args.dist_mode:
        dist.barrier()
    ckpt_dir = os.path.join(output_dir, 'ckpt')
    if global_rank == 0:
        os.makedirs(ckpt_dir, exist_ok=True)
        common_utils.backup_source_code(os.path.join(output_dir, 'code'))
        os.system('cp %s %s' % (args.cfg_file, output_dir))
        LazyConfig.save(cfg, os.path.join(output_dir, "config.yaml"))
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

    # log
    logger.info("Command line arguments: " + str(args))
    with open(args.cfg_file, "r", encoding="utf-8") as f:
        content = f.read()
    logger.info("Contents of args.config_file={}:\n{}".format(args.cfg_file, content))

    # trainer
    args.local_rank = local_rank
    args.global_rank = global_rank
    args.ckpt_dir = ckpt_dir
    if 'trainer' in cfg:
        cfg.trainer.args = args
        cfg.trainer.cfg = cfg
        cfg.trainer.logger = logger
        cfg.trainer.tb_writer = tb_writer
        model_trainer = instantiate(cfg.trainer)
    else:
        model_trainer = Trainer(args, cfg, logger, tb_writer)

    tbar = tqdm.trange(model_trainer.last_epoch + 1, model_trainer.cfg.runtime_params.train_epochs + 1,
                       desc='epochs', dynamic_ncols=True, disable=(local_rank != 0),
                       bar_format='{l_bar}{bar}{r_bar}\n')
    # train loop
    for current_epoch in tbar:
        model_trainer.train(current_epoch, tbar)
        model_trainer.save_ckpt(current_epoch)
        if current_epoch % model_trainer.cfg.runtime_params.eval_period == 0:
            model_trainer.evaluate(current_epoch)


if __name__ == '__main__':
    main()
