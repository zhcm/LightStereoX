# @Time    : 2025/2/27 18:07
# @Author  : zhangchenming
import os
import re
import numpy as np
from glob import glob
from PIL import Image
from collections import defaultdict
from .dataset_template import SequenceDatasetTemplate


class Isprs(SequenceDatasetTemplate):
    def __init__(self, data_root_path, augmentations, logger, sample_len):
        super().__init__(data_root_path, augmentations, logger)
        self.sample_len = sample_len

        all_seqs = sorted(glob(os.path.join(data_root_path, '*/*/*')))
        for each_seqs in all_seqs:
            left_images = sorted(glob(os.path.join(each_seqs, 'image_left/*_left.png')))
            self._append_sample(left_images)

    def _append_sample(self, left_images):
        seq_len = len(left_images)
        for ref_idx in range(0, seq_len - self.sample_len):

            sample = {'image': {'left': [], 'right': []},
                      'disparity': {'left': [], 'right': []}
                      }

            for idx in range(ref_idx, ref_idx + self.sample_len):
                sample["image"]['left'].append(left_images[idx])
                sample["image"]['right'].append(left_images[idx].replace('image_left', 'image_right').replace('_left.png', '_right.png'))
                sample["disparity"]['left'].append(left_images[idx].replace('image_left', 'depth_left').replace('_left.png', '_left_depth.npy'))
                sample["disparity"]['right'].append(left_images[idx].replace('image_left', 'depth_right').replace('_left.png', '_right_depth.npy'))

            self.sample_list.append(sample)

    def __getitem__(self, index):
        sample = self.sample_list[index]
        sample_size = len(sample["image"]["left"])

        output = defaultdict(list)
        output_keys = ["img", "disp", "valid_disp", "mask"]
        for key in output_keys:
            output[key] = [[] for _ in range(sample_size)]

        for i in range(sample_size):
            for cam in ["left", "right"]:
                img = Image.open(sample["image"][cam][i])
                img = np.array(img).astype(np.uint8)
                img = img[..., :3]
                output["img"][i].append(img)

                depth = np.load(sample["disparity"][cam][i])
                disp = (80.0 / depth).astype(np.float32)

                valid_disp = disp < 512
                disp = np.array(disp).astype(np.float32)
                disp = np.stack([-disp, np.zeros_like(disp)], axis=-1)  # [h, w, 2]
                output["disp"][i].append(disp)
                output["valid_disp"][i].append(valid_disp)  # [h, w] bool

        if self.augmentations is not None:
            for t in self.augmentations:
                output = t(output)

        for i in range(sample_size):
            for cam in (0, 1):
                disp = output["disp"][i][cam][..., 0]
                valid_disp = (np.abs(disp) < 512) & (disp != 0)
                output["valid_disp"][i][cam] = valid_disp.astype(np.float32)
                output["disp"][i][cam] = np.expand_dims(disp, axis=0)

        res = self.format_output(output)
        # {
        #     'img': [num_frames, 2(l&r), 3(c), h, w],
        #     'disp': [num_frames, 2(l&r), 1(c), h, w],
        #     'valid_disp': [num_frames, 2(l&r), h, w],
        #  }
        return res
