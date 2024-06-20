# @Time    : 2024/1/22 20:15
# @Author  : zhangchenming
import random
import torch
import numpy as np
import cv2
from torchvision.transforms.functional import normalize
from PIL import Image
from torchvision.transforms import ColorJitter, functional


class GetValidDisp(object):
    def __init__(self, config):
        self.max_disp = config.MAX_DISP

    def __call__(self, sample):
        disp = sample['disp']
        disp[disp > self.max_disp] = 0
        disp[disp < 0] = 0
        sample.update({
            'disp': disp,
        })
        if 'disp_right' in sample.keys():
            disp_right = sample['disp_right']
            disp_right[disp_right > self.max_disp] = 0
            disp_right[disp_right < 0] = 0
            sample.update({
                'disp_right': disp_right
            })

        return sample


class ShiftRandomCrop(object):
    # 右图，向右移动，视差变大
    def __init__(self, config):
        self.size = config.SIZE
        self.shift_range = config.SHIFT

    def __call__(self, sample):
        shift = random.randint(self.shift_range[0], self.shift_range[1])
        crop_height, crop_width = self.size
        height, width = sample['left'].shape[:2]  # (H, W, 3)
        crop_height = min(height, crop_height)
        crop_width = min(width - 2 * abs(shift), crop_width)

        y1 = random.randint(0, height - crop_height)
        y2 = y1 + crop_height
        x1 = random.randint(abs(shift), width - crop_width - abs(shift))
        x2 = x1 + crop_width

        img1 = sample['left']
        img2 = sample['right']
        disp = sample['disp']

        img1 = img1[y1: y2, x1: x2]
        img2 = img2[y1: y2, x1 + shift: x2 + shift]
        disp = disp[y1: y2, x1: x2] + shift

        sample['left'] = img1
        sample['right'] = img2
        sample['disp'] = disp

        return sample


class TestCrop(object):
    def __init__(self, config):
        self.size = config.SIZE

    def __call__(self, sample):
        h, w = sample['left'].shape[:2]
        crop_h, crop_w = self.size
        crop_h = min(h, crop_h)
        crop_w = min(w, crop_w)

        for k in sample.keys():
            sample[k] = sample[k][h - crop_h: h, w - crop_w: w]
        return sample


class CropOrPad(object):
    def __init__(self, config):
        self.size = config.SIZE

    def __call__(self, sample):
        h, w = sample['left'].shape[:2]
        th, tw = self.size
        if th > h or tw > w:
            # pad the arrays with zeros to the desired size
            pad_left = 0
            pad_right = tw - w
            pad_top = th - h
            pad_bottom = 0
            for k in sample.keys():
                if k in ['left', 'right']:
                    pad_width = np.array([[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
                    sample[k] = np.pad(sample[k], pad_width, 'edge')
                elif k in ['disp', 'disp_right', 'occ_mask', 'occ_mask_right']:
                    pad_width = np.array([[pad_top, pad_bottom], [pad_left, pad_right]])
                    sample[k] = np.pad(sample[k], pad_width, 'constant', constant_values=0)
        else:
            for k in sample.keys():
                if k in ['left', 'right', 'disp', 'disp_right', 'occ_mask', 'occ_mask_right']:
                    sample[k] = sample[k][h - th:h, w - tw: w]
        return sample


class DivisiblePad(object):
    def __init__(self, config):
        self.by = config.BY

    def __call__(self, sample):
        h, w = sample['left'].shape[:2]
        if h % self.by != 0:
            pad_top = h + self.by - h % self.by - h
        else:
            pad_top = 0
        if w % self.by != 0:
            pad_right = w + self.by - w % self.by - w
        else:
            pad_right = 0
        pad_left = 0
        pad_bottom = 0

        # apply pad for left, right, disp image, and occ mask
        for k in sample.keys():
            if k in ['left', 'right']:
                pad_width = np.array([[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
                sample[k] = np.pad(sample[k], pad_width, 'edge')
            elif k in ['disp', 'disp_right', 'occ_mask', 'occ_mask_right']:
                pad_width = np.array([[pad_top, pad_bottom], [pad_left, pad_right]])
                sample[k] = np.pad(sample[k], pad_width, 'constant', constant_values=0)
        sample['pad'] = [pad_top, pad_right, 0, 0]
        return sample


"""
FLOWAUGMENTOR START
"""


class AdjustGamma(object):

    def __init__(self, gamma_min, gamma_max, gain_min=1.0, gain_max=1.0):
        self.gamma_min, self.gamma_max, self.gain_min, self.gain_max = gamma_min, gamma_max, gain_min, gain_max

    def __call__(self, sample):
        gain = random.uniform(self.gain_min, self.gain_max)
        gamma = random.uniform(self.gamma_min, self.gamma_max)
        return functional.adjust_gamma(sample, gamma, gain)


class FlowAugmentor(object):
    def __init__(self, config):
        # spatial augmentation params
        self.crop_size = config.SIZE
        self.min_scale = config.MIN_SCALE
        self.max_scale = config.MAX_SCALE
        self.spatial_aug_prob = 1.0
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

        # flip augmentation params
        self.yjitter = False
        self.do_flip = config.DO_FLIP
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        saturation_range = (0.6, 1.4)
        gamma = [1, 1, 1, 1]

        # photometric augmentation params
        self.photo_aug = Compose(
            [ColorJitter(brightness=0.4, contrast=0.4, saturation=saturation_range, hue=0.5 / 3.14),
             AdjustGamma(*gamma)])
        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = 0.5

    def color_transform(self, img1, img2):
        """ Photometric augmentation """

        # asymmetric
        if np.random.rand() < self.asymmetric_color_aug_prob:
            img1 = np.array(self.photo_aug(Image.fromarray(img1.astype(np.uint8))), dtype=np.uint8)
            img2 = np.array(self.photo_aug(Image.fromarray(img2.astype(np.uint8))), dtype=np.uint8)

        # symmetric
        else:
            image_stack = np.concatenate([img1, img2], axis=0).astype(np.uint8)
            image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
            img1, img2 = np.split(image_stack, 2, axis=0)

        return img1, img2

    def eraser_transform(self, img1, img2, bounds=None):
        """ Occlusion augmentation """
        if bounds is None:
            bounds = [50, 100]
        ht, wd = img1.shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(bounds[0], bounds[1])
                dy = np.random.randint(bounds[0], bounds[1])
                img2[y0:y0 + dy, x0:x0 + dx, :] = mean_color

        return img1, img2

    def spatial_transform(self, img1, img2, flow):
        # randomly sample scale
        ht, wd = img1.shape[:2]
        min_scale = np.maximum(
            (self.crop_size[0] + 8) / float(ht),
            (self.crop_size[1] + 8) / float(wd))

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = scale
        scale_y = scale
        if np.random.rand() < self.stretch_prob:
            scale_x *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
            scale_y *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)

        scale_x = np.clip(scale_x, min_scale, None)
        scale_y = np.clip(scale_y, min_scale, None)

        if np.random.rand() < self.spatial_aug_prob:
            # rescale the images
            img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow = cv2.resize(flow, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)

            flow = flow * [scale_x, scale_y]

        if self.do_flip:
            if np.random.rand() < self.h_flip_prob and self.do_flip == 'hf':  # h-flip
                img1 = img1[:, ::-1]
                img2 = img2[:, ::-1]
                flow = flow[:, ::-1] * [-1.0, 1.0]

            if np.random.rand() < self.h_flip_prob and self.do_flip == 'h':  # h-flip for stereo
                tmp = img1[:, ::-1]
                img1 = img2[:, ::-1]
                img2 = tmp

            if np.random.rand() < self.v_flip_prob and self.do_flip == 'v':  # v-flip
                img1 = img1[::-1, :]
                img2 = img2[::-1, :]
                flow = flow[::-1, :] * [1.0, -1.0]

        if self.yjitter:
            y0 = np.random.randint(2, img1.shape[0] - self.crop_size[0] - 2)
            x0 = np.random.randint(2, img1.shape[1] - self.crop_size[1] - 2)

            y1 = y0 + np.random.randint(-2, 2 + 1)
            img1 = img1[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
            img2 = img2[y1:y1 + self.crop_size[0], x0:x0 + self.crop_size[1]]
            flow = flow[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]

        else:
            y0 = np.random.randint(0, img1.shape[0] - self.crop_size[0])
            x0 = np.random.randint(0, img1.shape[1] - self.crop_size[1])

            img1 = img1[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
            img2 = img2[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
            flow = flow[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]

        return img1, img2, flow

    def __call__(self, sample):
        img1 = sample['left']
        img2 = sample['right']
        disp = sample['disp']
        flow = np.stack([disp, np.zeros_like(disp)], axis=-1)

        img1, img2 = self.color_transform(img1, img2)
        img1, img2 = self.eraser_transform(img1, img2)
        img1, img2, flow = self.spatial_transform(img1, img2, flow)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        sample['left'] = img1
        sample['right'] = img2
        sample['disp'] = flow[:1][0]
        return sample


class ColorTransform(object):
    def __init__(self, config):
        self.config = config
        saturation_range = (0.6, 1.4)
        gamma = [1, 1, 1, 1]
        self.photo_aug = Compose(
            [ColorJitter(brightness=0.4, contrast=0.4, saturation=saturation_range, hue=0.5 / 3.14),
             AdjustGamma(*gamma)])
        self.asymmetric_color_aug_prob = 0.2

    def __call__(self, sample):
        img1 = sample['left']
        img2 = sample['right']
        # asymmetric
        if np.random.rand() < self.asymmetric_color_aug_prob:
            img1 = np.array(self.photo_aug(Image.fromarray(img1.astype(np.uint8))), dtype=np.uint8)
            img2 = np.array(self.photo_aug(Image.fromarray(img2.astype(np.uint8))), dtype=np.uint8)

        # symmetric
        else:
            image_stack = np.concatenate([img1, img2], axis=0).astype(np.uint8)
            image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
            img1, img2 = np.split(image_stack, 2, axis=0)

        sample['left'] = img1
        sample['right'] = img2

        return sample


class EraserTransform(object):
    def __init__(self, config):
        self.config = config
        self.eraser_aug_prob = 0.5

    def __call__(self, sample):
        bounds = [50, 100]
        img1 = sample['left']
        img2 = sample['right']

        ht, wd = img1.shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(bounds[0], bounds[1])
                dy = np.random.randint(bounds[0], bounds[1])
                img2[y0:y0 + dy, x0:x0 + dx, :] = mean_color

        sample['left'] = img1
        sample['right'] = img2
        return sample


class ScaleTransform(object):
    def __init__(self, config):
        self.config = config
        self.crop_size = config.SIZE
        self.min_scale = config.MIN_SCALE
        self.max_scale = config.MAX_SCALE
        self.stretch_prob = 0.8
        self.max_stretch = 0.2
        self.spatial_aug_prob = 1.0

    def __call__(self, sample):
        img1 = sample['left']
        img2 = sample['right']
        disp = sample['disp']

        ht, wd = img1.shape[:2]
        min_scale = np.maximum((self.crop_size[0] + 8) / float(ht), (self.crop_size[1] + 8) / float(wd))

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = scale
        scale_y = scale
        if np.random.rand() < self.stretch_prob:
            scale_x *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
            scale_y *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)

        scale_x = np.clip(scale_x, min_scale, None)
        scale_y = np.clip(scale_y, min_scale, None)

        if np.random.rand() < self.spatial_aug_prob:
            img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            disp = cv2.resize(disp, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            disp = disp * scale_x

        sample['left'] = img1
        sample['right'] = img2
        sample['disp'] = disp
        return sample


class FlipTransform(object):
    def __init__(self, config):
        self.config = config
        self.flip_type = config.FLIP_TYPE
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

    def __call__(self, sample):
        img1 = sample['left']
        img2 = sample['right']
        disp = sample['disp']
        disp_right = sample['disp_right']

        if np.random.rand() < self.h_flip_prob and self.flip_type == 'hf':  # h-flip
            img1 = np.ascontiguousarray(img1[:, ::-1])
            img2 = np.ascontiguousarray(img2[:, ::-1])
            disp = np.ascontiguousarray(disp[:, ::-1] * -1.0)

        if np.random.rand() < self.h_flip_prob and self.flip_type == 'h':  # h-flip for stereo
            tmp = np.ascontiguousarray(img1[:, ::-1])
            img1 = np.ascontiguousarray(img2[:, ::-1])
            disp = np.ascontiguousarray(disp_right[:, ::-1])
            img2 = tmp

        if np.random.rand() < self.v_flip_prob and self.flip_type == 'v':  # v-flip
            img1 = np.ascontiguousarray(img1[::-1, :])
            img2 = np.ascontiguousarray(img2[::-1, :])
            disp = np.ascontiguousarray(disp[::-1, :])

        sample['left'] = img1
        sample['right'] = img2
        sample['disp'] = disp
        return sample


class CropTransform(object):
    def __init__(self, config):
        self.config = config
        self.crop_size = config.SIZE
        self.base_size = config.SIZE

    def __call__(self, sample):
        img1 = sample['left']
        img2 = sample['right']
        disp = sample['disp']

        y0 = np.random.randint(0, img1.shape[0] - self.crop_size[0])
        x0 = np.random.randint(0, img1.shape[1] - self.crop_size[1])

        img1 = img1[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        img2 = img2[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        disp = disp[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]

        sample['left'] = img1
        sample['right'] = img2
        sample['disp'] = disp
        return sample


class FormatTransform(object):
    def __init__(self, config):
        self.config = config

    def __call__(self, sample):
        img1 = sample['left']
        img2 = sample['right']
        disp = sample['disp']

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        disp = torch.from_numpy(disp).float()

        sample['left'] = img1
        sample['right'] = img2
        sample['disp'] = disp
        return sample


"""
FLOWAUGMENTOR STOP
"""

"""
SPARSEFLOWAUGMENTOR START
"""
class SparseFlowAugmentor(object):
    def __init__(self, config):

        self.crop_size = config.SIZE
        self.min_scale = config.MIN_SCALE
        self.max_scale = config.MAX_SCALE
        self.do_flip = config.DO_FLIP

        self.spatial_aug_prob = 0.8
        self.stretch_prob = 0.8
        self.max_stretch = 0.2
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        saturation_range = [0.7, 1.3]
        gamma = [1, 1, 1, 1]
        # photometric augmentation params
        self.photo_aug = Compose(
            [ColorJitter(brightness=0.3, contrast=0.3, saturation=saturation_range, hue=0.3 / 3.14),
             AdjustGamma(*gamma)])
        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = 0.5

    def color_transform(self, img1, img2):
        image_stack = np.concatenate([img1, img2], axis=0)
        image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
        img1, img2 = np.split(image_stack, 2, axis=0)
        return img1, img2

    def eraser_transform(self, img1, img2):
        ht, wd = img1.shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(50, 100)
                dy = np.random.randint(50, 100)
                img2[y0:y0 + dy, x0:x0 + dx, :] = mean_color

        return img1, img2

    def resize_sparse_flow_map(self, flow, valid, fx=1.0, fy=1.0):
        ht, wd = flow.shape[:2]
        coords = np.meshgrid(np.arange(wd), np.arange(ht))
        coords = np.stack(coords, axis=-1)

        coords = coords.reshape(-1, 2).astype(np.float32)
        flow = flow.reshape(-1, 2).astype(np.float32)
        valid = valid.reshape(-1).astype(np.float32)

        coords0 = coords[valid >= 1]
        flow0 = flow[valid >= 1]

        ht1 = int(round(ht * fy))
        wd1 = int(round(wd * fx))

        coords1 = coords0 * [fx, fy]
        flow1 = flow0 * [fx, fy]

        xx = np.round(coords1[:, 0]).astype(np.int32)
        yy = np.round(coords1[:, 1]).astype(np.int32)

        v = (xx > 0) & (xx < wd1) & (yy > 0) & (yy < ht1)
        xx = xx[v]
        yy = yy[v]
        flow1 = flow1[v]

        flow_img = np.zeros([ht1, wd1, 2], dtype=np.float32)
        valid_img = np.zeros([ht1, wd1], dtype=np.int32)

        flow_img[yy, xx] = flow1
        valid_img[yy, xx] = 1

        return flow_img, valid_img

    def spatial_transform(self, img1, img2, flow, valid):
        # randomly sample scale

        ht, wd = img1.shape[:2]
        min_scale = np.maximum(
            (self.crop_size[0] + 1) / float(ht),
            (self.crop_size[1] + 1) / float(wd))

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = np.clip(scale, min_scale, None)
        scale_y = np.clip(scale, min_scale, None)

        if np.random.rand() < self.spatial_aug_prob:
            # rescale the images
            img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow, valid = self.resize_sparse_flow_map(flow, valid, fx=scale_x, fy=scale_y)

        if self.do_flip:
            if np.random.rand() < self.h_flip_prob and self.do_flip == 'hf':  # h-flip
                img1 = img1[:, ::-1]
                img2 = img2[:, ::-1]
                flow = flow[:, ::-1] * [-1.0, 1.0]

            if np.random.rand() < self.h_flip_prob and self.do_flip == 'h':  # h-flip for stereo
                tmp = img1[:, ::-1]
                img1 = img2[:, ::-1]
                img2 = tmp

            if np.random.rand() < self.v_flip_prob and self.do_flip == 'v':  # v-flip
                img1 = img1[::-1, :]
                img2 = img2[::-1, :]
                flow = flow[::-1, :] * [1.0, -1.0]

        margin_y = 20
        margin_x = 50

        y0 = np.random.randint(0, img1.shape[0] - self.crop_size[0] + margin_y)
        x0 = np.random.randint(-margin_x, img1.shape[1] - self.crop_size[1] + margin_x)

        y0 = np.clip(y0, 0, img1.shape[0] - self.crop_size[0])
        x0 = np.clip(x0, 0, img1.shape[1] - self.crop_size[1])

        img1 = img1[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        img2 = img2[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        flow = flow[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        valid = valid[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]

        return img1, img2, flow, valid

    def __call__(self, sample):
        img1 = sample['left']
        img2 = sample['right']
        disp = sample['disp']

        valid = sample['disp'] > 0.0

        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)
        img1 = img1[..., :3]
        img2 = img2[..., :3]

        flow = np.stack([disp, np.zeros_like(disp)], axis=-1)

        img1, img2 = self.color_transform(img1, img2)
        img1, img2 = self.eraser_transform(img1, img2)  # 与EraserTransform相同
        img1, img2, flow, valid = self.spatial_transform(img1, img2, flow, valid)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()
        valid = torch.from_numpy(valid).float()

        sample['left'] = img1
        sample['right'] = img2
        sample['disp'] = flow[:1][0]
        sample['flow'] = flow[:1][0]
        sample['valid'] = valid
        return sample


class SparseColorTransform(object):
    def __init__(self, config):
        self.config = config
        saturation_range = [0.7, 1.3]
        gamma = [1, 1, 1, 1]
        self.photo_aug = Compose(
            [ColorJitter(brightness=0.3, contrast=0.3, saturation=saturation_range, hue=0.3 / 3.14),
             AdjustGamma(*gamma)])

    def __call__(self, sample):
        img1 = sample['left']
        img2 = sample['right']

        image_stack = np.concatenate([img1, img2], axis=0).astype(np.uint8)
        image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
        img1, img2 = np.split(image_stack, 2, axis=0)

        sample['left'] = img1
        sample['right'] = img2

        return sample


class SparseScaleTransform(object):
    def __init__(self, config):
        self.config = config
        self.crop_size = config.SIZE
        self.min_scale = config.MIN_SCALE
        self.max_scale = config.MAX_SCALE
        self.spatial_aug_prob = 0.8
    def __call__(self, sample):
        img1 = sample['left']
        img2 = sample['right']

        disp = sample['disp']
        valid = sample['disp'] > 0.0

        flow = np.stack([disp, np.zeros_like(disp)], axis=-1)

        ht, wd = img1.shape[:2]
        min_scale = np.maximum((self.crop_size[0] + 1) / float(ht), (self.crop_size[1] + 1) / float(wd))

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = np.clip(scale, min_scale, None)
        scale_y = np.clip(scale, min_scale, None)

        if np.random.rand() < self.spatial_aug_prob:
            # rescale the images
            img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow, _ = self.resize_sparse_flow_map(flow, valid, fx=scale_x, fy=scale_y)

        sample['left'] = img1
        sample['right'] = img2
        sample['disp'] = flow[:, :, 0]

        return sample

    @staticmethod
    def resize_sparse_flow_map(flow, valid, fx=1.0, fy=1.0):
        ht, wd = flow.shape[:2]
        coords = np.meshgrid(np.arange(wd), np.arange(ht))
        coords = np.stack(coords, axis=-1)

        coords = coords.reshape(-1, 2).astype(np.float32)
        flow = flow.reshape(-1, 2).astype(np.float32)
        valid = valid.reshape(-1).astype(np.float32)

        coords0 = coords[valid >= 1]
        flow0 = flow[valid >= 1]

        ht1 = int(round(ht * fy))
        wd1 = int(round(wd * fx))

        coords1 = coords0 * [fx, fy]
        flow1 = flow0 * [fx, fy]

        xx = np.round(coords1[:, 0]).astype(np.int32)
        yy = np.round(coords1[:, 1]).astype(np.int32)

        v = (xx > 0) & (xx < wd1) & (yy > 0) & (yy < ht1)
        xx = xx[v]
        yy = yy[v]
        flow1 = flow1[v]

        flow_img = np.zeros([ht1, wd1, 2], dtype=np.float32)
        valid_img = np.zeros([ht1, wd1], dtype=np.int32)

        flow_img[yy, xx] = flow1
        valid_img[yy, xx] = 1

        return flow_img, valid_img


class SparseCropTransform(object):
    def __init__(self, config):
        self.config = config
        self.crop_size = config.SIZE

    def __call__(self, sample):
        margin_y = 20
        margin_x = 50

        img1 = sample['left']
        img2 = sample['right']
        disp = sample['disp']

        y0 = np.random.randint(0, img1.shape[0] - self.crop_size[0] + margin_y)
        x0 = np.random.randint(-margin_x, img1.shape[1] - self.crop_size[1] + margin_x)

        y0 = np.clip(y0, 0, img1.shape[0] - self.crop_size[0])
        x0 = np.clip(x0, 0, img1.shape[1] - self.crop_size[1])

        img1 = img1[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        img2 = img2[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        disp = disp[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]

        sample['left'] = img1
        sample['right'] = img2
        sample['disp'] = disp
        return sample
