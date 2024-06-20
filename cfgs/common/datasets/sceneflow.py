# @Time    : 2024/6/9 18:13
# @Author  : zhangchenming
from stereo.config.lazy import LazyCall
from stereo.datasets.sceneflow_dataset import SceneFlowDataset

data_root_path = '/mnt/nas/algorithm/chenming.zhang/dataset/SceneFlow'

train = LazyCall(SceneFlowDataset)(
    data_root_path=data_root_path,
    split_file='./data/sceneflow/sceneflow_finalpass_train.txt',
    augmentations=None,
    return_right_disp=False
)

val = LazyCall(SceneFlowDataset)(
    data_root_path=data_root_path,
    split_file='./data/sceneflow/sceneflow_finalpass_test.txt',
    augmentations=None,
    return_right_disp=False
)
