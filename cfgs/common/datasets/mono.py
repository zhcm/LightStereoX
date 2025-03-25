# @Time    : 2024/6/9 18:13
# @Author  : zhangchenming
import os
from stereo.config.lazy import LazyCall
from stereo.datasets.mono_dataset import MonoDataset
from stereo.datasets.realfill_dataset import RealfillDataset

from cfgs.common.runtime_params import data_root_dir

data_root_path = os.path.join(data_root_dir, 'depthAnythingData')

train_gl = LazyCall(MonoDataset)(
    data_root_path=data_root_path,
    split_file='/baai-cwm-1/baai_cwm_ml/public_data/scenes/depthAnythingData/txt_path_files/GoogleLandmarks.txt',
    augmentations=None,
)  # 4976922 4976143

train_bdd = LazyCall(MonoDataset)(
    data_root_path=data_root_path,
    split_file='/baai-cwm-1/baai_cwm_ml/public_data/scenes/depthAnythingData/txt_path_files/bdd100k.txt',
    augmentations=None,
)  # 100000 99717

train_bdd_realfill = LazyCall(RealfillDataset)(
    data_root_path=data_root_path,
    split_file='/baai-cwm-1/baai_cwm_ml/public_data/scenes/depthAnythingData/txt_path_files/bdd100K_realfill.txt',
    augmentations=None,
)  # 100000

train_objects365_realfill = LazyCall(RealfillDataset)(
    data_root_path=data_root_path,
    split_file='/baai-cwm-1/baai_cwm_ml/algorithm/xianda.guo/code/Wrl/OpenStereo/objects365/object365_path_txt/leftRightDisp.txt',
    augmentations=None,
)

train_21k = LazyCall(MonoDataset)(
    data_root_path=data_root_path,
    split_file='/baai-cwm-1/baai_cwm_ml/public_data/scenes/depthAnythingData/txt_path_files/imagenet21k_resize.txt',
    augmentations=None,
)  # 13145689 13150758

train_object365 = LazyCall(MonoDataset)(
    data_root_path=data_root_path,
    split_file='/baai-cwm-1/baai_cwm_ml/public_data/scenes/depthAnythingData/txt_path_files/objects365_resize.txt',
    augmentations=None,
)  # 2168460 2007977

train_lsun = LazyCall(MonoDataset)(
    data_root_path=data_root_path,
    split_file='/baai-cwm-1/baai_cwm_ml/public_data/scenes/depthAnythingData/txt_path_files/lsun_resize.txt',
    augmentations=None,
)  # 9898281 9898284

train_sa1b = LazyCall(MonoDataset)(
    data_root_path=data_root_path,
    split_file='/baai-cwm-1/baai_cwm_ml/public_data/scenes/depthAnythingData/txt_path_files/sa-1b.txt',
    augmentations=None,
)

train_openimage = LazyCall(MonoDataset)(
    data_root_path=data_root_path,
    split_file='/baai-cwm-1/baai_cwm_ml/public_data/scenes/depthAnythingData/txt_path_files/openimage.txt',
    augmentations=None,
)  # 1910098

train_place365 = LazyCall(MonoDataset)(
    data_root_path=data_root_path,
    split_file='/baai-cwm-1/baai_cwm_ml/public_data/scenes/depthAnythingData/txt_path_files/places365.txt',
    augmentations=None,
)  # 2168460
