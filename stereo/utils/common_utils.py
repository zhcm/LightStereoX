# @Time    : 2023/8/28 22:28
# @Author  : zhangchenming
import os
import random
import shutil
import numpy as np
import torch
import logging
import matplotlib.pyplot as plt


def color_map_tensorboard(disp_gt, pred, disp_max=192):
    cm = plt.get_cmap('plasma')

    disp_gt = disp_gt.detach().data.cpu().numpy()
    pred = pred.detach().data.cpu().numpy()
    error_map = np.abs(pred - disp_gt)

    disp_gt = np.clip(disp_gt, a_min=0, a_max=disp_max)
    pred = np.clip(pred, a_min=0, a_max=disp_max)

    gt_tmp = 255.0 * disp_gt / disp_max
    pred_tmp = 255.0 * pred / disp_max
    error_map_tmp = 255.0 * error_map / np.max(error_map)

    gt_tmp = cm(gt_tmp.astype('uint8'))
    pred_tmp = cm(pred_tmp.astype('uint8'))
    error_map_tmp = cm(error_map_tmp.astype('uint8'))

    gt_tmp = np.transpose(gt_tmp[:, :, :3], (2, 0, 1))
    pred_tmp = np.transpose(pred_tmp[:, :, :3], (2, 0, 1))
    error_map_tmp = np.transpose(error_map_tmp[:, :, :3], (2, 0, 1))

    color_disp_c = np.concatenate((gt_tmp, pred_tmp, error_map_tmp), axis=1)
    color_disp_c = torch.from_numpy(color_disp_c)

    return color_disp_c


def set_random_seed(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)


def create_logger(log_file=None, rank=0):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO if rank == 0 else logging.ERROR)
    formatter = logging.Formatter('%(asctime)s  %(levelname)5s  %(message)s')

    console = logging.StreamHandler()
    console.setLevel(logging.INFO if rank == 0 else logging.ERROR)
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_file is not None and rank == 0:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setLevel(logging.INFO if rank == 0 else logging.ERROR)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def backup_source_code(backup_dir):
    # 子文件夹下的同名也会被忽略
    ignore_hidden = shutil.ignore_patterns(
        ".idea", ".git*", "*pycache*",
        "cfgs", "data", "output")

    if os.path.exists(backup_dir):
        shutil.rmtree(backup_dir)

    shutil.copytree('.', backup_dir, ignore=ignore_hidden)


def save_checkpoint(model, optimizer, scheduler, scaler, is_dist, output_dir):
    if is_dist:
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()

    optim_state = optimizer.state_dict()
    scheduler_state = scheduler.state_dict()
    scaler_state = scaler.state_dict()

    torch.save(model_state, os.path.join(output_dir, 'pytorch_model.bin'))
    torch.save(optim_state, os.path.join(output_dir, 'optimizer.bin'))
    torch.save(scheduler_state, os.path.join(output_dir, 'scheduler.bin'))
    torch.save(scaler_state, os.path.join(output_dir, 'scaler.pt'))


def freeze_bn(module):
    """Freeze the batch normalization layers."""
    for m in module.modules():
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.eval()
    return module


def load_params_from_file(model, filename, device, logger, strict=True):
    pretrained_state_dict = torch.load(filename, map_location=device)
    state_dict = model.state_dict()

    unused_state_dict = {}
    update_state_dict = {}
    unupdate_state_dict = {}
    for key, val in pretrained_state_dict.items():
        if key in state_dict and state_dict[key].shape == val.shape:
            update_state_dict[key] = val
        else:
            unused_state_dict[key] = val
    for key in state_dict:
        if key not in update_state_dict:
            unupdate_state_dict[key] = state_dict[key]

    if strict:
        model.load_state_dict(update_state_dict)
    else:
        state_dict.update(update_state_dict)
        model.load_state_dict(state_dict)

    message = 'Unused weight: '
    for key, val in unused_state_dict.items():
        message += str(key) + ':' + str(val.shape) + ', '
    if logger:
        logger.info(message)
    else:
        print(message)

    message = 'Not updated weight: '
    for key, val in unupdate_state_dict.items():
        message += str(key) + ':' + str(val.shape) + ', '
    if logger:
        logger.info(message)
    else:
        print(message)


def write_tensorboard(tb_writer, tb_info, step):
    for k, v in tb_info.items():
        module_name = k.split('/')[0]
        writer_module = getattr(tb_writer, 'add_' + module_name)
        board_name = k.replace(module_name + "/", '')
        v = v.detach() if torch.is_tensor(v) else v
        if module_name == 'image' and v.dim() == 2:
            writer_module(board_name, v, step, dataformats='HW')
        else:
            writer_module(board_name, v, step)
