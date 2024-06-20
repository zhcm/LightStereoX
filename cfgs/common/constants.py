from omegaconf import OmegaConf

constants = OmegaConf.create(
    dict(
        imagenet_rgb_mean=[123.675, 116.28, 103.53],
        imagenet_rgb_std=[58.395, 57.12, 57.375],
        rgb_mean=[127.5, 127.5, 127.5],
        rgb_std=[127.5, 127.5, 127.5]
    )
)
