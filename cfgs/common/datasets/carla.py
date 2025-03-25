# @Time    : 2024/6/9 18:13
# @Author  : zhangchenming
import os
from stereo.config.lazy import LazyCall
from stereo.datasets.carla_dataset import CarlaDataset

from cfgs.common.runtime_params import data_root_dir

data_root_path = os.path.join(data_root_dir, 'StereoFromCarlaV2')

train = LazyCall(CarlaDataset)(
    data_root_path=data_root_path,
    split_file='./data/Carla/CarlaStereo.txt',
    augmentations=None,
)  # 552050

weather_train = LazyCall(CarlaDataset)(
    data_root_path=data_root_path,
    split_file='./data/Carla/WeatherStereo.txt',
    augmentations=None,
)
# (1080, 1920, 3)
