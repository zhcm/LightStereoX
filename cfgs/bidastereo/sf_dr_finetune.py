# @Time    : 2025/2/16 08:14
# @Author  : zhangchenming
import os
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from stereo.config.lazy import LazyCall, LazyConfig
from stereo.sequence_datasets.utils import stereo_trans
from stereo.datasets import build_dataloader
from stereo.modeling.models.bidastereo.bidastereo import BiDAStereo
from stereo.modeling.models.bidastereo.get_params import get_all_model_params

from stereo.solver.build import ClipGradNorm
from stereo.solver.trainer_bida import BIDATrainer

from cfgs.common.runtime_params import runtime_params, ckpt_root_dir
from cfgs.common.constants import constants

augmentations = {
    'train': [
        LazyCall(stereo_trans.StereoColorJitter)(brightness=[0.6, 1.4], contrast=[0.6, 1.4],
                                                 saturation=[0.0, 1.4], hue=[-0.5 / 3.14, 0.5 / 3.14],
                                                 asymmetric_prob=0.2),
        LazyCall(stereo_trans.RandomErase)(prob=0.5, max_time=2, bounds=[50, 100]),
        LazyCall(stereo_trans.RandomScale)(crop_size=[256, 512], min_pow_scale=-0.2, max_pow_scale=0.4, scale_prob=1.0, stretch_prob=0.8),
        LazyCall(stereo_trans.RandomCrop)(crop_size=[256, 512], y_jitter=True),
        LazyCall(stereo_trans.NormalizeImage)(mean=constants.standard_rgb_mean, std=constants.standard_rgb_std)
    ],
    'val': [
        LazyCall(stereo_trans.DivisiblePad)(divisor=32, mode='round'),
        LazyCall(stereo_trans.NormalizeImage)(mean=constants.standard_rgb_mean, std=constants.standard_rgb_std)
    ]
}

sceneflow = LazyConfig.load('cfgs/common/sequence_datasets/sceneflow.py')
sceneflow.train_clean.augmentations = augmentations['train']
sceneflow.train_clean.sample_len = 5
sceneflow.train_final.augmentations = augmentations['train']
sceneflow.train_final.sample_len = 5

dynamic = LazyConfig.load('cfgs/common/sequence_datasets/dynamic_replica.py')
dynamic.train.augmentations = augmentations['train']
dynamic.train.sample_len = 5

val_data = LazyConfig.load('cfgs/common/sequence_datasets/sintel_clean.py')
val_data.train_clean.augmentations = augmentations['val']

# dataloader
batch_size_per_gpu = 1
train_loader = LazyCall(build_dataloader)(
    is_dist=None,
    all_dataset=[sceneflow.train_clean, sceneflow.train_final, dynamic.train],
    batch_size=batch_size_per_gpu,
    shuffle=True,
    workers=16,
    pin_memory=True,
    drop_last=True)

val_loader = LazyCall(build_dataloader)(
    is_dist=None,
    all_dataset=[val_data.train_clean],
    batch_size=batch_size_per_gpu * 2,
    shuffle=False,
    workers=16,
    pin_memory=True,
    drop_last=False)

model = LazyCall(BiDAStereo)(train_iters=10, eval_iters=20)

# optim
lr = 0.0004
optimizer = LazyCall(AdamW)(
    params=LazyCall(get_all_model_params)(model=None),
    lr=lr,
    weight_decay=1.0e-05,
    eps=1.0e-08)

trainer = LazyCall(BIDATrainer)(args=None, cfg=None, logger=None, tb_writer=None)

# (22390 + 17088 + 8720) * 2 + 45402 = 141798
# 141798 / 8 = 17725
# 60000 / 17725 = 4
runtime_params.save_root_dir = os.path.join(ckpt_root_dir, 'output/SequenceSceneFlowDataset/BiDAStereo')
runtime_params.max_iter = 60000
runtime_params.eval_period = 100
runtime_params.find_unused_parameters = False
runtime_params.freeze_bn = True
runtime_params.pretrained_model = os.path.join(ckpt_root_dir, 'output/SequenceSceneFlowDataset/BiDAStereo/sf_dr_pretrain/ckpt/epoch_10/pytorch_model.bin')

# scheduler
scheduler = LazyCall(OneCycleLR)(optimizer=None, max_lr=lr, total_steps=runtime_params.max_iter+100, pct_start=0.01,
                                 cycle_momentum=False, anneal_strategy="linear")

clip_grad = LazyCall(ClipGradNorm)(max_norm=1.0)
