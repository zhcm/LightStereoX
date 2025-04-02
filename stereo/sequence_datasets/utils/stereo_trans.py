# @Time    : 2024/6/21 05:41
# @Author  : zhangchenming
import numpy as np
import cv2

from PIL import Image
from torchvision.transforms import ColorJitter


class StereoColorJitter(object):
    def __init__(self, brightness, contrast, saturation, hue, asymmetric_prob=0.0):
        self.brightness = list(brightness)
        self.contrast = list(contrast)
        self.saturation = list(saturation)
        self.hue = list(hue)
        self.asymmetric_prob = asymmetric_prob
        self.color_jitter = ColorJitter(brightness=self.brightness,
                                        contrast=self.contrast,
                                        saturation=self.saturation,
                                        hue=self.hue)

    def __call__(self, sample):
        seq_img = sample['img']
        # asymmetric
        if np.random.rand() < self.asymmetric_prob:
            for i in range(len(seq_img)):
                for cam in (0, 1):
                    seq_img[i][cam] = np.array(self.color_jitter(Image.fromarray(seq_img[i][cam])), dtype=np.uint8)

        # symmetric
        else:
            image_stack = np.concatenate([seq_img[i][cam] for i in range(len(seq_img)) for cam in (0, 1)], axis=0)
            image_stack = np.array(self.color_jitter(Image.fromarray(image_stack)), dtype=np.uint8)
            split = np.split(image_stack, len(seq_img) * 2, axis=0)
            for i in range(len(seq_img)):
                seq_img[i][0] = split[2 * i]
                seq_img[i][1] = split[2 * i + 1]

        sample['img'] = seq_img
        return sample


class RandomErase(object):
    def __init__(self, prob, max_time, bounds):
        self.prob = prob
        self.max_time = max_time
        self.bounds = bounds

    def __call__(self, sample):
        seq_img = sample['img']
        h, w = seq_img[0][0].shape[:2]
        for i in range(len(seq_img)):
            for cam in (0, 1):
                if np.random.rand() < self.prob:
                    mean_color = np.mean(seq_img[0][0].reshape(-1, 3), axis=0)
                    for _ in range(np.random.randint(1, self.max_time + 1)):
                        x0 = np.random.randint(0, w)
                        y0 = np.random.randint(0, h)
                        dx = np.random.randint(self.bounds[0], self.bounds[1])
                        dy = np.random.randint(self.bounds[0], self.bounds[1])
                        seq_img[i][cam][y0:y0 + dy, x0:x0 + dx, :] = mean_color

        sample['img'] = seq_img
        return sample


class RandomScale(object):
    def __init__(self, crop_size, min_pow_scale, max_pow_scale, scale_prob, stretch_prob):
        self.crop_size = crop_size
        self.min_pow_scale = min_pow_scale
        self.max_pow_scale = max_pow_scale
        self.scale_prob = scale_prob
        self.stretch_prob = stretch_prob
        self.stretch = [-0.2, 0.2]

    def __call__(self, sample):
        seq_img = sample['img']
        seq_disp = sample['disp']
        h, w = seq_img[0][0].shape[:2]

        floor_scale = max((self.crop_size[0] + 8) / h, (self.crop_size[1] + 8) / w)
        scale = 2 ** np.random.uniform(self.min_pow_scale, self.max_pow_scale)
        scale_x = scale
        scale_y = scale
        if np.random.rand() < self.stretch_prob:
            scale_x *= 2 ** np.random.uniform(self.stretch[0], self.stretch[1])
            scale_y *= 2 ** np.random.uniform(self.stretch[0], self.stretch[1])

        scale_x = max(scale_x, floor_scale)
        scale_y = max(scale_y, floor_scale)

        if np.random.rand() < self.scale_prob:
            for i in range(len(seq_img)):
                for cam in (0, 1):
                    seq_img[i][cam] = cv2.resize(seq_img[i][cam], None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)

                if len(seq_disp[i]) == 1:
                    seq_disp[i][0] = cv2.resize(seq_disp[i][0], None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
                    seq_disp[i][0] = seq_disp[i][0] * [scale_x, scale_y]
                    seq_disp[i][0] = seq_disp[i][0][..., 0:1].astype(np.float32)
                elif len(seq_disp[i]) == 2:
                    seq_disp[i][0] = cv2.resize(seq_disp[i][0], None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
                    seq_disp[i][0] = seq_disp[i][0] * [scale_x, scale_y]
                    seq_disp[i][0] = seq_disp[i][0][..., 0:1].astype(np.float32)
                    seq_disp[i][1] = cv2.resize(seq_disp[i][1], None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
                    seq_disp[i][1] = seq_disp[i][1] * [scale_x, scale_y]
                    seq_disp[i][1] = seq_disp[i][1][..., 0:1].astype(np.float32)

        sample['img'] = seq_img
        sample['disp'] = seq_disp
        return sample


class RandomCrop(object):
    def __init__(self, crop_size, y_jitter=False):
        self.crop_size = crop_size
        self.base_size = crop_size
        self.y_jitter = y_jitter

    def __call__(self, sample):
        seq_img = sample['img']
        seq_disp = sample['disp']
        crop_height, crop_width = self.crop_size
        height, width = seq_img[0][0].shape[:2]  # (H, W, 3)
        if crop_width > width or crop_height > height:
            return sample

        n_pixels = 2 if self.y_jitter else 0
        y0 = np.random.randint(n_pixels, height - crop_height - n_pixels)
        x0 = np.random.randint(n_pixels, width - crop_width - n_pixels)

        for i in range(len(seq_img)):
            y1 = y0 + np.random.randint(-n_pixels, n_pixels + 1)
            seq_img[i][0] = seq_img[i][0][y0: y0 + crop_height, x0: x0 + crop_width]
            seq_img[i][1] = seq_img[i][1][y1: y1 + crop_height, x0: x0 + crop_width]

            if len(seq_disp[i]) == 2:
                seq_disp[i][0] = seq_disp[i][0][y0: y0 + crop_height, x0: x0 + crop_width]
                seq_disp[i][1] = seq_disp[i][1][y1: y1 + crop_height, x0: x0 + crop_width]
            elif len(seq_disp[i]) == 1:
                seq_disp[i][0] = seq_disp[i][0][y0: y0 + crop_height, x0: x0 + crop_width]

        sample['img'] = seq_img
        sample['disp'] = seq_disp
        return sample


class NormalizeImage(object):
    def __init__(self, mean, std):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, sample):
        seq_img = sample['img']
        for i in range(len(seq_img)):
            for cam in (0, 1):
                seq_img[i][cam] = (seq_img[i][cam] - self.mean) / self.std
                seq_img[i][cam] = seq_img[i][cam].transpose((2, 0, 1))

        sample['img'] = seq_img
        return sample


class DivisiblePad(object):
    def __init__(self, divisor, mode='tr'):
        self.divisor = divisor
        self.mode = mode

    def __call__(self, sample):
        seq_img = sample['img']
        h, w = seq_img[0][0].shape[:2]
        if h % self.divisor != 0:
            pad_h = self.divisor - h % self.divisor
        else:
            pad_h = 0
        if w % self.divisor != 0:
            pad_w = self.divisor - w % self.divisor
        else:
            pad_w = 0

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
            raise Exception('no DivisiblePad mode')

        for i in range(len(seq_img)):
            for cam in (0, 1):
                pad_width = np.array([[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
                seq_img[i][cam] = np.pad(seq_img[i][cam], pad_width, 'edge')

        sample['pad'] = np.array([pad_top, pad_right, pad_bottom, pad_left])
        sample['img'] = seq_img
        return sample
