# @Time    : 2025/2/15 22:27
# @Author  : zhangchenming
import os
import numpy as np

from PIL import Image
from glob import glob
from collections import defaultdict

from ..datasets.utils.readpfm import readpfm
from .dataset_template import SequenceDatasetTemplate


class SequenceSceneFlowDataset(SequenceDatasetTemplate):
    def __init__(self, data_root_path, augmentations, logger, dataset_type, sample_len):
        super().__init__(data_root_path, augmentations, logger)
        self.data_root_path = data_root_path
        self.dataset_type = dataset_type
        self.sample_len = sample_len

        self._add_things("TRAIN")
        self._add_monkaa()
        self._add_driving()

    def _append_sample(self, images, disparities):
        seq_len = len(images["left"])
        for ref_idx in range(0, seq_len - self.sample_len):
            # 正向
            sample = {'image': {'left': [], 'right': []},
                      'disparity': {'left': [], 'right': []}
                      }
            for cam in ["left", "right"]:
                for idx in range(ref_idx, ref_idx + self.sample_len):
                    sample["image"][cam].append(images[cam][idx])
                    sample["disparity"][cam].append(disparities[cam][idx])
            self.sample_list.append(sample)

            # 反向
            sample = {'image': {'left': [], 'right': []},
                      'disparity': {'left': [], 'right': []}
                      }
            for cam in ["left", "right"]:
                for idx in range(ref_idx, ref_idx + self.sample_len):
                    sample["image"][cam].append(images[cam][seq_len - idx - 1])
                    sample["disparity"][cam].append(disparities[cam][seq_len - idx - 1])
            self.sample_list.append(sample)

    def _add_things(self, split="TRAIN"):
        original_length = len(self.sample_list)
        root = os.path.join(self.data_root_path, "FlyingThings3D")
        image_dirs = defaultdict(list)
        disparity_dirs = defaultdict(list)

        for cam in ["left", "right"]:
            image_dirs[cam] = sorted(glob(os.path.join(root, self.dataset_type, split, f"*/*/{cam}/")))
            disparity_dirs[cam] = [path.replace(self.dataset_type, "disparity") for path in image_dirs[cam]]

        num_seq = len(image_dirs["left"])  # 序列数量
        num = 0
        for seq_idx in range(num_seq):
            images, disparities = defaultdict(list), defaultdict(list)
            for cam in ["left", "right"]:
                images[cam] = sorted(glob(os.path.join(image_dirs[cam][seq_idx], "*.png")))
                disparities[cam] = sorted(glob(os.path.join(disparity_dirs[cam][seq_idx], "*.pfm")))
            num = num + len(images["left"])  # 图片数量
            self._append_sample(images, disparities)

        assert len(self.sample_list) > 0, "No samples found"
        self.logger.info(f"Added {len(self.sample_list) - original_length} from FlyingThings {self.dataset_type}")

    def _add_monkaa(self):
        original_length = len(self.sample_list)
        root = os.path.join(self.data_root_path, "Monkaa")
        image_dirs = defaultdict(list)
        disparity_dirs = defaultdict(list)

        for cam in ["left", "right"]:
            image_dirs[cam] = sorted(glob(os.path.join(root, self.dataset_type, f"*/{cam}/")))
            disparity_dirs[cam] = [path.replace(self.dataset_type, "disparity") for path in image_dirs[cam]]

        num_seq = len(image_dirs["left"])

        for seq_idx in range(num_seq):
            images, disparities = defaultdict(list), defaultdict(list)
            for cam in ["left", "right"]:
                images[cam] = sorted(glob(os.path.join(image_dirs[cam][seq_idx], "*.png")))
                disparities[cam] = sorted(glob(os.path.join(disparity_dirs[cam][seq_idx], "*.pfm")))

            self._append_sample(images, disparities)

        assert len(self.sample_list) > 0, "No samples found"
        self.logger.info(f"Added {len(self.sample_list) - original_length} from Monkaa {self.dataset_type}")

    def _add_driving(self):
        original_length = len(self.sample_list)
        root = os.path.join(self.data_root_path, "Driving")
        image_dirs = defaultdict(list)
        disparity_dirs = defaultdict(list)

        for cam in ["left", "right"]:
            image_dirs[cam] = sorted(glob(os.path.join(root, self.dataset_type, f"*/*/*/{cam}/")))
            disparity_dirs[cam] = [path.replace(self.dataset_type, "disparity") for path in image_dirs[cam]]

        num_seq = len(image_dirs["left"])
        for seq_idx in range(num_seq):
            images, disparities = defaultdict(list), defaultdict(list)
            for cam in ["left", "right"]:
                images[cam] = sorted(glob(os.path.join(image_dirs[cam][seq_idx], "*.png")))
                disparities[cam] = sorted(glob(os.path.join(disparity_dirs[cam][seq_idx], "*.pfm")))

            self._append_sample(images, disparities)

        assert len(self.sample_list) > 0, "No samples found"
        self.logger.info(f"Added {len(self.sample_list) - original_length} from Driving {self.dataset_type}")

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

                disp = readpfm(sample["disparity"][cam][i])[0].astype(np.float32)  # [h, w]
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
