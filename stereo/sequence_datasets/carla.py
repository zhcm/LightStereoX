# @Time    : 2025/2/27 18:07
# @Author  : zhangchenming
import os
import re
import numpy as np
from glob import glob
from PIL import Image
from collections import defaultdict
from .dataset_template import SequenceDatasetTemplate


class SequenceCarlaDataset(SequenceDatasetTemplate):
    def __init__(self, data_root_path, augmentations, logger, sample_len):
        super().__init__(data_root_path, augmentations, logger)
        self.sample_len = sample_len

        all_seqs = sorted(glob(os.path.join(data_root_path, '*/*')))
        for each_seqs in all_seqs:
            if 'weather' in each_seqs:
                continue
            left_images = sorted(glob(os.path.join(each_seqs, 'left/rgb/*.jpg')))
            self._append_sample(left_images)

    def _append_sample(self, left_images):
        seq_len = len(left_images)
        for ref_idx in range(0, seq_len - self.sample_len):
            for each_baseline in ['baseline_010', 'baseline_054', 'baseline_100', 'baseline_200', 'baseline_300']:
                sample = {'image': {'left': [], 'right': []},
                          'disparity': {'left': []}
                          }

                for idx in range(ref_idx, ref_idx + self.sample_len):
                    sample["image"]['left'].append(left_images[idx])
                    sample["image"]['right'].append(left_images[idx].replace('left', each_baseline))
                    sample["disparity"]['left'].append(left_images[idx].replace('left/rgb', each_baseline + '/depth').replace('.jpg', '.png'))

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

            right_path = sample["image"]['right'][i]
            if 'baseline_010' in right_path:
                baseline = 10.0
            elif 'baseline_054' in right_path:
                baseline = 54.0
            elif 'baseline_100' in right_path:
                baseline = 100.0
            elif 'baseline_200' in right_path:
                baseline = 200.0
            elif 'baseline_300' in right_path:
                baseline = 300.0
            f_pix = 1385.64
            depth = np.array(Image.open(sample["disparity"]['left'][i]), dtype=np.float32)  # cm
            disp = baseline * f_pix / (depth + 1e-6)

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
