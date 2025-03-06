# @Time    : 2024/6/11 03:14
# @Author  : zhangchenming
from omegaconf import OmegaConf
import configparser

runtime_params = OmegaConf.create(
    dict(
        fix_random_seed=True,

        find_unused_parameters=False,
        freeze_bn=False,
        use_sync_bn=True,
        mixed_precision=False,

        save_root_dir='./output',
        pretrained_model='',
        resume_from_ckpt=-1,

        max_ckpt_save_num=30,
        train_epochs=0,
        log_period=10,
        train_visualization=True,

        eval_period=1,
        eval_visualization=True,
        eval_max_disp=192
    )
)

config_file = "cfgs/env.ini"
config = configparser.ConfigParser()
config.read(config_file)

env = config['environment']['env'].strip()
if env == "zy":
    project_root_dir = '/baai-cwm-1/baai_cwm_ml/algorithm/xianda.guo/code/chm/code/LightStereoX'
    data_root_dir = '/baai-cwm-1/baai_cwm_ml/public_data/scenes/stereo'
    ckpt_root_dir = '/baai-cwm-nas/algorithm/xianda.guo/checkpoints/chm/LightStereoX'
elif env == "vol":
    project_root_dir = '/file_system/vepfs/algorithm/chenming.zhang/code/LightStereoX'
    data_root_dir = '/file_system/vepfs/public_data/stereo'
    ckpt_root_dir = '/file_system/nas/algorithm/chenming.zhang/checkpoints/LightStereoX'
else:
    raise "工作集群为定义(zy or vol)"




