# @Time    : 2024/6/21 05:41
# @Author  : zhangchenming
import random
import numpy as np
import cv2

from PIL import Image
from torchvision.transforms import ColorJitter


class RandomCrop(object):
    def __init__(self, crop_size, y_jitter=False):
        self.crop_size = crop_size
        self.base_size = crop_size
        self.y_jitter = y_jitter

    def __call__(self, sample):
        crop_height, crop_width = self.crop_size
        height, width = sample['left'].shape[:2]  # (H, W, 3)
        if crop_width > width or crop_height > height:
            return sample

        n_pixels = 2 if (self.y_jitter and np.random.rand() < 0.5) else 0
        y1 = np.random.randint(n_pixels, height - crop_height - n_pixels + 1)
        x1 = np.random.randint(0, width - crop_width + 1)
        y2 = y1 + np.random.randint(-n_pixels, n_pixels + 1)

        for k in sample.keys():
            if k in ['pad']:
                continue
            if k in ['right', 'disp_right', 'occ_mask_right']:
                sample[k] = sample[k][y2: y2 + crop_height, x1: x1 + crop_width]
            else:
                sample[k] = sample[k][y1: y1 + crop_height, x1: x1 + crop_width]

        return sample


class KittiRandomCrop(object):
    def __init__(self, crop_size, y_jitter=False):
        self.crop_size = crop_size
        self.base_size = crop_size
        self.y_jitter = y_jitter

    def __call__(self, sample):
        crop_height, crop_width = self.crop_size
        height, width = sample['left'].shape[:2]  # (H, W, 3)
        if crop_width > width or crop_height > height:
            return sample

        margin_y = 20
        margin_x = 50

        y1 = np.random.randint(0, height - crop_height + margin_y)
        x1 = np.random.randint(-margin_x, width - crop_width + margin_x)

        y1 = np.clip(y1, 0, height - crop_height)
        x1 = np.clip(x1, 0, width - crop_width)

        for k in sample.keys():
            sample[k] = sample[k][y1: y1 + crop_height, x1: x1 + crop_width]

        return sample


class ShiftRandomCrop(object):
    # crop的位置向左偏，视差变大, shift=old-new,shift>0,向右
    def __init__(self, crop_size, shift):
        self.crop_size = crop_size
        self.shift = shift

    def __call__(self, sample):
        crop_height, crop_width = self.crop_size
        height, width = sample['left'].shape[:2]  # (H, W, 3)
        if crop_width > width or crop_height > height:
            return sample

        y1 = np.random.randint(0, height - crop_height + 1)
        x1 = np.random.randint(0, width - crop_width + 1)

        if self.shift >= 0:
            x2_right = min(x1 + self.shift + crop_width, width)
            x1_right = x2_right - crop_width
        else:
            x1_right = max(x1 + self.shift, 0)

        for k in sample.keys():
            if k in ['right']:
                sample[k] = sample[k][y1: y1 + crop_height, x1_right: x1_right + crop_width]
            else:
                sample[k] = sample[k][y1: y1 + crop_height, x1: x1 + crop_width]

        sample['disp'] = sample['disp'] + x1_right - x1

        return sample


class ConstantCrop(object):
    def __init__(self, crop_size, mode='tl'):
        self.size = crop_size
        self.mode = 'tl'

    def __call__(self, sample):
        h, w = sample['left'].shape[:2]
        crop_h, crop_w = self.size
        crop_h = min(h, crop_h)
        crop_w = min(w, crop_w)

        for k in sample.keys():
            sample[k] = sample[k][h - crop_h:, w - crop_w:]
        return sample


class NormalizeImage(object):
    def __init__(self, mean, std):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, sample):
        for k in sample.keys():
            if k in ['left', 'right', 'left_fl', 'right_fl']:
                sample[k] = (sample[k] - self.mean) / self.std
                sample[k] = sample[k].transpose((2, 0, 1))
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
            if k in ['left', 'right', 'left_fl', 'right_fl']:
                pad_width = np.array([[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
                sample[k] = np.pad(sample[k], pad_width, 'edge')

            elif k in ['disp', 'disp_right', 'occ_mask', 'occ_mask_right', 'bump_mask', 'height_map']:
                pad_width = np.array([[pad_top, pad_bottom], [pad_left, pad_right]])
                sample[k] = np.pad(sample[k], pad_width, 'constant', constant_values=0)

        sample['pad'] = np.array([pad_top, pad_right, pad_bottom, pad_left])

        return sample


class DivisiblePad(object):
    def __init__(self, divisor, mode='tr'):
        self.divisor = divisor
        self.mode = mode

    def __call__(self, sample):
        h, w = sample['left'].shape[:2]
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


class CropOrPad(object):
    def __init__(self, size):
        self.size = size
        self.crop_fn = ConstantCrop(crop_size=size, mode='tl')
        self.pad_fn = ConstantPad(target_size=size, mode='tr')

    def __call__(self, sample):
        h, w = sample['left'].shape[:2]
        th, tw = self.size
        if th > h or tw > w:
            sample = self.pad_fn(sample)
        else:
            sample = self.crop_fn(sample)

        return sample


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
        img1 = sample['left']
        img2 = sample['right']
        # asymmetric
        if np.random.rand() < self.asymmetric_prob:
            img1 = np.array(self.color_jitter(Image.fromarray(img1.astype(np.uint8))), dtype=np.uint8)
            img2 = np.array(self.color_jitter(Image.fromarray(img2.astype(np.uint8))), dtype=np.uint8)
        # symmetric
        else:
            image_stack = np.concatenate([img1, img2], axis=0).astype(np.uint8)
            image_stack = np.array(self.color_jitter(Image.fromarray(image_stack)), dtype=np.uint8)
            img1, img2 = np.split(image_stack, 2, axis=0)

        sample['left'] = img1
        sample['right'] = img2

        return sample


class RandomErase(object):
    def __init__(self, prob, max_time, bounds):
        self.prob = prob
        self.max_time = max_time
        self.bounds = bounds

    def __call__(self, sample):
        img1 = sample['left']
        img2 = sample['right']

        h, w = img1.shape[:2]
        occ_mask_2 = np.zeros((h, w), dtype=bool)
        if np.random.rand() < self.prob:
            mean_color = np.mean(img2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, self.max_time + 1)):
                x0 = np.random.randint(0, w)
                y0 = np.random.randint(0, h)
                dx = np.random.randint(self.bounds[0], self.bounds[1])
                dy = np.random.randint(self.bounds[0], self.bounds[1])
                img2[y0:y0 + dy, x0:x0 + dx, :] = mean_color
                occ_mask_2[y0:y0 + dy, x0:x0 + dx] = True

        sample['left'] = img1
        sample['right'] = img2
        sample['occ_mask_2'] = occ_mask_2
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
        h, w = sample['left'].shape[:2]

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
            for k in sample.keys():
                if k in ['left', 'right']:
                    sample[k] = cv2.resize(sample[k], None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)

                elif k in ['disp', 'disp_right']:
                    sample[k] = cv2.resize(sample[k], None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
                    sample[k] = sample[k] * scale_x

                elif k in ['occ_mask']:
                    uint8_array = sample[k].astype(np.uint8)
                    resized_array = cv2.resize(uint8_array, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_NEAREST)
                    sample[k] = resized_array.astype(bool)

        return sample


class RandomSparseScale(object):
    def __init__(self, crop_size, min_pow_scale, max_pow_scale, prob):
        self.crop_size = crop_size
        self.min_pow_scale = min_pow_scale
        self.max_pow_scale = max_pow_scale
        self.prob = prob

    def __call__(self, sample):
        h, w = sample['left'].shape[:2]
        floor_scale = max((self.crop_size[0] + 1) / h, (self.crop_size[1] + 1) / w)
        scale = 2 ** np.random.uniform(self.min_pow_scale, self.max_pow_scale)
        scale = max(scale, floor_scale)

        # valid_img = sample['disp'] > 0.0
        if np.random.rand() < self.prob:
            for k in sample.keys():
                if k in ['left', 'right']:
                    sample[k] = cv2.resize(sample[k], dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                elif k in ['disp', 'disp_right']:
                    sample[k], valid_img = self.sparse_disp_map_reisze(sample[k], fx=scale, fy=scale)
                elif k in ['occ_mask', 'occ_mask_2']:
                    sample[k] = cv2.resize(sample[k].astype(np.float32), None, fx=scale, fy=scale,
                                           interpolation=cv2.INTER_NEAREST) > 0.5
                elif k in ['super_pixel_label']:
                    sample[k] = cv2.resize(sample[k], None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

        sample['valid'] = valid_img > 0
        return sample

    @staticmethod
    def sparse_disp_map_reisze(disp, fx=1.0, fy=1.0):
        h, w = disp.shape[:2]
        h_new = round(h * fy)
        w_new = round(w * fx)

        coords = np.meshgrid(np.arange(w), np.arange(h))
        coords = np.stack(coords, axis=-1)  # [h, w, 2] 坐标(x, y)
        coords = coords.reshape(-1, 2).astype(np.float32)  # [h*w, 2] 坐标(x, y)

        disp = disp.reshape(-1).astype(np.float32)  # [h*w,]
        valid = disp > 0.0
        coords = coords[valid]  # disp > 0 的坐标
        disp = disp[valid]  # disp > 0 的值
        coords = coords * [fx, fy]  # resize 后的坐标
        disp = disp * fx  # risize 后的值

        coords_x = np.round(coords[:, 0]).astype(np.int32)
        coords_y = np.round(coords[:, 1]).astype(np.int32)
        v = (coords_x > 0) & (coords_x < w_new) & (coords_y > 0) & (coords_y < h_new)

        coords_x = coords_x[v]
        coords_y = coords_y[v]
        disp = disp[v]

        resized_disp = np.zeros([h_new, w_new], dtype=np.float32)
        resized_disp[coords_y, coords_x] = disp

        valid_img = np.zeros([h_new, w_new], dtype=np.int32)
        valid_img[coords_y, coords_x] = 1

        return resized_disp, valid_img
