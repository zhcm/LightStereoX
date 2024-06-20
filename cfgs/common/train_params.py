# @Time    : 2024/6/11 03:14
# @Author  : zhangchenming
from omegaconf import OmegaConf

train_params = OmegaConf.create(
    dict(
        fix_random_seed=True,
        mixed_precision=False,
        find_unused_parameters=False,
        freeze_bn=False,
        use_sync_bn=True,
        save_root_dir='./output',
        train_epochs=0,
        pretrained_model='',
        resume_from_ckpt=-1,
        max_ckpt_save_num=30,
        log_period=10,
        eval_period=1
    )
)
