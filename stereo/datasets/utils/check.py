# @Time    : 2024/6/22 22:58
# @Author  : zhangchenming
import torch
import numpy as np

from torchvision.transforms.functional import normalize


class TransposeImage(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        sample['left'] = sample['left'].transpose((2, 0, 1))
        sample['right'] = sample['right'].transpose((2, 0, 1))
        return sample


class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        for k in sample.keys():
            if isinstance(sample[k], np.ndarray):
                sample[k] = torch.from_numpy(sample[k].copy()).to(torch.float32)
        return sample


class NormalizeImage(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        sample['left'] = normalize(sample['left'] / 255.0, self.mean, self.std)
        sample['right'] = normalize(sample['right'] / 255.0, self.mean, self.std)
        return sample
