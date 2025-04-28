# @Time    : 2025/2/27 18:07
# @Author  : zhangchenming
import os
import cv2
import numpy as np
from glob import glob
from PIL import Image
from collections import defaultdict
from .dataset_template import SequenceDatasetTemplate
from stereo.datasets.mono import WarpDataset
from stereo.datasets.mono_dataset import transfer_color
import random


class SequenceKittiDataset(SequenceDatasetTemplate):
    def __init__(self, data_root_path, augmentations, logger, sample_len):
        super().__init__(data_root_path, augmentations, logger)
        self.sample_len = sample_len
        self.warp = WarpDataset()

        all_seqs = sorted(glob(os.path.join(data_root_path, '*')))
        for each_seqs in all_seqs:
            if '.txt' in each_seqs:
                continue
            left_images = sorted(glob(os.path.join(each_seqs, 'left/*.png')))
            self._append_sample(left_images)

    def _append_sample(self, left_images):
        seq_len = len(left_images)
        for ref_idx in range(0, seq_len - self.sample_len):
            sample = {'image': {'left': []},
                      'disparity': {'left': []}
                      }

            for idx in range(ref_idx, ref_idx + self.sample_len):
                sample["image"]['left'].append(left_images[idx])
                sample["disparity"]['left'].append(left_images[idx].replace('left/', 'depth/'))

            self.sample_list.append(sample)

    def __getitem__(self, index):
        sample = self.sample_list[index]
        sample_size = len(sample["image"]["left"])

        output = defaultdict(list)
        output_keys = ["img", "disp", "valid_disp", "mask"]
        for key in output_keys:
            output[key] = [[] for _ in range(sample_size)]

        for i in range(sample_size):
            background_path = random.choice(sample["image"]["left"])
            background_image = Image.open(background_path).convert('RGB')
            loaded_disparity = cv2.imread(sample["disparity"]['left'][i], cv2.IMREAD_UNCHANGED)
            loaded_disparity = loaded_disparity.astype(np.float32) / 100
            inputs = {'left_image': Image.open(sample["image"]["left"][i]).convert('RGB'),
                      'background': background_image,
                      'loaded_disparity': loaded_disparity[:, :, 0]}
            inputs = self.warp.prepare_sizes(inputs)
            inputs['background'] = transfer_color(np.array(inputs['background']), np.array(inputs['left_image']))
            inputs['disparity'] = self.warp.process_disparity(inputs['loaded_disparity'], max_disparity_range=(192, 192))
            projection_disparity = inputs['disparity']
            right_image = self.warp.project_image(inputs['left_image'], projection_disparity, inputs['background'])

            left_image = np.array(inputs['left_image'], dtype=np.float32)
            output["img"][i].append(left_image.astype(np.uint8))
            output["img"][i].append(right_image.astype(np.uint8))

            disp = inputs['disparity'].astype(np.float32)
            valid_disp = disp < 512
            disp = np.array(disp).astype(np.float32)
            disp = np.stack([-disp, np.zeros_like(disp)], axis=-1)  # [h, w, 2]
            output["disp"][i].append(disp)
            output["valid_disp"][i].append(valid_disp)  # [h, w] bool

        if self.augmentations is not None:
            for t in self.augmentations:
                output = t(output)

        for i in range(sample_size):
            disp = output["disp"][i][0][..., 0]
            valid_disp = (np.abs(disp) < 512) & (disp != 0)
            output["valid_disp"][i][0] = valid_disp.astype(np.float32)
            output["disp"][i][0] = np.expand_dims(disp, axis=0)

        res = self.format_output(output)
        # {
        #     'img': [num_frames, 2(l&r), 3(c), h, w],
        #     'disp': [num_frames, 1(l), 1(c), h, w],
        #     'valid_disp': [num_frames, 1(l), h, w],
        #  }
        return res
