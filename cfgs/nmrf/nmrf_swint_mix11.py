# @Time    : 2024/10/8 02:37
# @Author  : zhangchenming
import os
from torch.optim.lr_scheduler import OneCycleLR

from stereo.config.lazy import LazyCall, LazyConfig
from stereo.datasets.utils import stereo_trans
from stereo.datasets import build_dataloader

from stereo.modeling.models.nmrf.backbone import create_backbone
from stereo.modeling.models.nmrf.DPN import DPN
from stereo.modeling.models.nmrf.NMRF import NMRF, Criterion
from stereo.modeling.models.nmrf.build_optimizer import build_optimizer, for_compatibility
from stereo.solver.build import ClipGradNorm

from cfgs.common.runtime_params import runtime_params, ckpt_root_dir
from cfgs.common.constants import constants

train_augmentations = [
    LazyCall(stereo_trans.StereoColorJitter)(brightness=[0.6, 1.4], contrast=[0.6, 1.4],
                                             saturation=[0.6, 1.4], hue=[-0.5 / 3.14, 0.5 / 3.14],
                                             asymmetric_prob=0.2),
    LazyCall(stereo_trans.RandomErase)(prob=0.5, max_time=2, bounds=[50, 100]),
    LazyCall(stereo_trans.RandomCrop)(crop_size=[352, 640]),
    LazyCall(stereo_trans.NormalizeImage)(mean=constants.imagenet_rgb_mean, std=constants.imagenet_rgb_std)
]

carla = LazyConfig.load('cfgs/common/datasets/carla.py')  # 552050
carla.train.augmentations = train_augmentations
carla.weather_train.augmentations = train_augmentations

dynamic = LazyConfig.load('cfgs/common/datasets/dynamic.py')  # 144900
dynamic.train.augmentations = train_augmentations

crestereo = LazyConfig.load('cfgs/common/datasets/crestereo.py')  # 200000
crestereo.train.augmentations = train_augmentations
crestereo.train.return_right_disp = False

fallingthings = LazyConfig.load('cfgs/common/datasets/fallingthings.py')  # 61500
fallingthings.train.augmentations = train_augmentations
fallingthings.train.return_right_disp = False

instereo2k = LazyConfig.load('cfgs/common/datasets/instereo2k.py')  # 2010
instereo2k.train.augmentations = train_augmentations
instereo2k.train.return_right_disp = False

tartanair = LazyConfig.load('cfgs/common/datasets/tartanair.py')  # 306637
tartanair.train.augmentations = train_augmentations

sintel = LazyConfig.load('cfgs/common/datasets/sintel.py')  # 1064
sintel.train.augmentations = train_augmentations

spring = LazyConfig.load('cfgs/common/datasets/spring.py')  # 5000
spring.train.augmentations = train_augmentations
spring.train.return_right_disp = False

virtualkitti2 = LazyConfig.load('cfgs/common/datasets/virtualkitti2.py')  # 21260
virtualkitti2.train.augmentations = train_augmentations
virtualkitti2.train.return_right_disp = False

argoverse = LazyConfig.load('cfgs/common/datasets/argoverse.py')
argoverse.train.augmentations = train_augmentations

unrealstereo4k = LazyConfig.load('cfgs/common/datasets/unrealstereo4k.py')
unrealstereo4k.train.augmentations = train_augmentations
unrealstereo4k.train.return_right_disp = False

# dataloader
batch_size_per_gpu = 2
train_loader = LazyCall(build_dataloader)(
    is_dist=None,
    all_dataset=[tartanair.train, carla.train, carla.weather_train, crestereo.train, spring.train, sintel.train, dynamic.train, fallingthings.train, instereo2k.train, virtualkitti2.train,
                 argoverse.train, unrealstereo4k.train],
    batch_size=batch_size_per_gpu,
    shuffle=True,
    workers=8,
    pin_memory=True,
    drop_last=True)

val_loader = LazyCall(build_dataloader)(
    is_dist=None,
    all_dataset=[sintel.train],
    batch_size=batch_size_per_gpu * 2,
    shuffle=False,
    workers=8,
    pin_memory=True,
    drop_last=False)

weight_dict = {'proposal_disp': 1,
               'init': 1,
               'loss_coarse_disp_0': 1.0,
               'loss_coarse_disp_1': 1.0,
               'loss_coarse_disp_2': 1.0,
               'loss_coarse_disp_3': 1.4,
               'loss_coarse_disp_4': 1.4,
               'loss_disp_5': 1.4,
               'loss_disp_6': 1.4,
               'loss_disp_7': 1.6,
               'loss_disp_8': 2.0,
               'loss_disp': 2.0}
criterion = LazyCall(Criterion)(weight_dict=weight_dict, max_disp=192, loss_type='L1')

model = LazyCall(NMRF)(
    backbone=LazyCall(create_backbone)(model_type='swin', norm_fn='instance', out_channels=128, drop_path=0.4),
    dpn=LazyCall(DPN)(cost_group=4, num_proposals=4, feat_dim=128, context_dim=64, num_prop_layers=5,
                      prop_embed_dim=128, mlp_ratio=4, split_size=1, prop_n_heads=4,
                      normalize_before=True),
    num_proposals=4,
    max_disp=320,
    num_infer_layers=5,
    num_refine_layers=5,
    infer_embed_dim=128,
    infer_n_heads=4,
    mlp_ratio=4,
    window_size=6,
    refine_window_size=4,
    return_intermediate=True,
    normalize_before=True,
    aux_loss=True,
    divis_by=32,
    compat=False,
    criterion=criterion)

# optim
lr = 0.0005 * batch_size_per_gpu / 2 * 2
optimizer = LazyCall(build_optimizer)(params=LazyCall(for_compatibility)(model=None), base_lr=lr)

# scheduler
scheduler = LazyCall(OneCycleLR)(optimizer=None, max_lr=lr, total_steps=-1, pct_start=0.05,
                                 cycle_momentum=False, anneal_strategy='cos')

clip_grad = LazyCall(ClipGradNorm)(max_norm=1.0)

runtime_params.save_root_dir = os.path.join(ckpt_root_dir, 'output/MixDataset/NMRF')
runtime_params.train_epochs = 1
runtime_params.eval_period = 10
runtime_params.pretrained_model = os.path.join(ckpt_root_dir, 'output/SceneFlowDataset/NMRF/swint/ckpt/epoch_67/pytorch_model.bin')
