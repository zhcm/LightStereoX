# @Time    : 2024/6/21 05:41
# @Author  : zhangchenming
import random
import numpy as np


class RandomCrop(object):
    def __init__(self, crop_size, y_jitter=False):
        self.crop_size = crop_size
        self.y_jitter = y_jitter

    def __call__(self, sample):
        crop_height, crop_width = self.crop_size
        height, width = sample['left'].shape[:2]  # (H, W, 3)
        if crop_width > width or crop_height > height:
            return sample

        n_pixels = 2 if (self.y_jitter and np.random.rand() < 0.5) else 0
        y1 = random.randint(n_pixels, height - crop_height - n_pixels)
        x1 = random.randint(0, width - crop_width)
        y2 = y1 + np.random.randint(-n_pixels, n_pixels + 1)

        for k in sample.keys():
            if k in ['right', 'disp_right']:
                sample[k] = sample[k][y2: y2 + crop_height, x1: x1 + crop_width]
            else:
                sample[k] = sample[k][y1: y1 + crop_height, x1: x1 + crop_width]

        return sample


class NormalizeImage(object):
    def __init__(self, mean, std):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, sample):
        sample['left'] = (sample['left'] - self.mean) / self.std
        sample['right'] = (sample['right'] - self.mean) / self.std

        sample['left'] = sample['left'].transpose((2, 0, 1))
        sample['right'] = sample['right'].transpose((2, 0, 1))
        return sample


class ConstantPad(object):
    def __init__(self, target_size, mode='tr'):
        self.size = target_size
        self.mode = mode

    def __call__(self, sample):
        h, w = sample['left'].shape[:2]
        th, tw = self.size
        pad_h = max(th - h, 0)
        pad_w = max(tw - w, 0)

        if self.mode == 'round':
            pad_top = pad_h // 2
            pad_right = pad_w // 2
            pad_bottom = pad_h - (pad_h // 2)
            pad_left = pad_w - (pad_w // 2)
        elif self.mode == 'tr':
            pad_top = pad_h
            pad_right = pad_w
            pad_bottom = 0
            pad_left = 0
        else:
            raise Exception('no ConstantPad mode')

        # apply pad for left, right, disp image, and occ mask
        for k in sample.keys():
            if k in ['left', 'right']:
                pad_width = np.array([[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
                sample[k] = np.pad(sample[k], pad_width, 'edge')

            elif k in ['disp', 'disp_right', 'occ_mask', 'occ_mask_right']:
                pad_width = np.array([[pad_top, pad_bottom], [pad_left, pad_right]])
                sample[k] = np.pad(sample[k], pad_width, 'constant', constant_values=0)

        sample['pad'] = np.array([pad_top, pad_right, pad_bottom, pad_left])

        return sample
