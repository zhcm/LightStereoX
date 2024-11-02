# @Time    : 2024/10/27 02:00
# @Author  : zhangchenming
import numpy as np
import random
from skimage.filters import gaussian, sobel
from scipy.interpolate import griddata


class WarpDataset:
    def __init__(self):
        self.max_disparity = 192
        self.process_width = self.feed_width + self.max_disparity
        self.xs, self.ys = np.meshgrid(np.arange(self.process_width), np.arange(self.feed_height))

    def transfer_color(self, target, source):
        target = target.astype(float) / 255
        source = source.astype(float) / 255

        target_means = target.mean(0).mean(0)
        target_stds = target.std(0).std(0)

        source_means = source.mean(0).mean(0)
        source_stds = source.std(0).std(0)

        target -= target_means
        target /= target_stds / source_stds
        target += source_means

        target = np.clip(target, 0, 1)
        target = (target * 255).astype(np.uint8)

        return target

    def process_disparity(self, disparity, max_disparity_range=(40, 196)):
        """ Depth predictions have arbitrary scale - need to convert to a pixel disparity"""

        disparity = disparity.copy()

        # make disparities positive
        min_disp = disparity.min()
        if min_disp < 0:
            disparity += np.abs(min_disp)

        if random.random() < 0.01:
            # make max warped disparity bigger than network max -> will be clipped to max disparity,
            # but will mean network is robust to disparities which are too big
            max_disparity_range = (self.max_disparity * 1.05, self.max_disparity * 1.15)

        disparity /= disparity.max()  # now 0-1

        scaling_factor = (max_disparity_range[0] + random.random() *
                          (max_disparity_range[1] - max_disparity_range[0]))
        disparity *= scaling_factor

        if not self.disable_sharpening:
            # now find disparity gradients and set to nearest - stop flying pixels
            edges = sobel(disparity) > 3
            disparity[edges] = 0
            mask = disparity > 0

            try:
                disparity = griddata(np.stack([self.ys[mask].ravel(), self.xs[mask].ravel()], 1),
                                     disparity[mask].ravel(), np.stack([self.ys.ravel(),
                                                                        self.xs.ravel()], 1),
                                     method='nearest').reshape(self.feed_height, self.process_width)
            except (ValueError, IndexError) as e:
                pass  # just return disparity

        return disparity

    def project_image(self, image, disp_map, background_image):

        image = np.array(image)
        background_image = np.array(background_image)

        # set up for projection
        warped_image = np.zeros_like(image).astype(float)
        warped_image = np.stack([warped_image] * 2, 0)
        pix_locations = self.xs - disp_map

        # find where occlusions are, and remove from disparity map
        mask = self.get_occlusion_mask(pix_locations)
        masked_pix_locations = pix_locations * mask - self.process_width * (1 - mask)

        # do projection - linear interpolate up to 1 pixel away
        weights = np.ones((2, self.feed_height, self.process_width)) * 10000

        for col in range(self.process_width - 1, -1, -1):
            loc = masked_pix_locations[:, col]
            loc_up = np.ceil(loc).astype(int)
            loc_down = np.floor(loc).astype(int)
            weight_up = loc_up - loc
            weight_down = 1 - weight_up

            mask = loc_up >= 0
            mask[mask] = \
                weights[0, np.arange(self.feed_height)[mask], loc_up[mask]] > weight_up[mask]
            weights[0, np.arange(self.feed_height)[mask], loc_up[mask]] = \
                weight_up[mask]
            warped_image[0, np.arange(self.feed_height)[mask], loc_up[mask]] = \
                image[:, col][mask] / 255.

            mask = loc_down >= 0
            mask[mask] = \
                weights[1, np.arange(self.feed_height)[mask], loc_down[mask]] > weight_down[mask]
            weights[1, np.arange(self.feed_height)[mask], loc_down[mask]] = weight_down[mask]
            warped_image[1, np.arange(self.feed_height)[mask], loc_down[mask]] = \
                image[:, col][mask] / 255.

        weights /= weights.sum(0, keepdims=True) + 1e-7  # normalise
        weights = np.expand_dims(weights, -1)
        warped_image = warped_image[0] * weights[1] + warped_image[1] * weights[0]
        warped_image *= 255.

        # now fill occluded regions with random background
        if not self.disable_background:
            warped_image[warped_image.max(-1) == 0] = background_image[warped_image.max(-1) == 0]

        warped_image = warped_image.astype(np.uint8)

        return warped_image

    def get_occlusion_mask(self, shifted):

        mask_up = shifted > 0
        mask_down = shifted > 0

        shifted_up = np.ceil(shifted)
        shifted_down = np.floor(shifted)

        for col in range(self.process_width - 2):
            loc = shifted[:, col:col + 1]  # keepdims
            loc_up = np.ceil(loc)
            loc_down = np.floor(loc)

            _mask_down = ((shifted_down[:, col + 2:] != loc_down) * (
            (shifted_up[:, col + 2:] != loc_down))).min(-1)
            _mask_up = ((shifted_down[:, col + 2:] != loc_up) * (
            (shifted_up[:, col + 2:] != loc_up))).min(-1)

            mask_up[:, col] = mask_up[:, col] * _mask_up
            mask_down[:, col] = mask_down[:, col] * _mask_down

        mask = mask_up + mask_down
        return mask
