# @Time    : 2025/2/15 22:27
# @Author  : zhangchenming
import os
import numpy as np

from PIL import Image
from glob import glob
from collections import defaultdict

from .dataset_template import SequenceDatasetTemplate


def disparity_reader(file_name):
    """Return disparity read from filename."""
    f_in = np.array(Image.open(file_name))
    d_r = f_in[:, :, 0].astype("float64")
    d_g = f_in[:, :, 1].astype("float64")
    d_b = f_in[:, :, 2].astype("float64")

    disp = d_r * 4 + d_g / (2 ** 6) + d_b / (2 ** 14)
    mask = np.array(Image.open(file_name.replace("disparities", "occlusions")))
    valid = (mask == 0) & (disp > 0)
    return disp, valid


class SequenceSintelDataset(SequenceDatasetTemplate):
    def __init__(self, data_root_path, augmentations, logger, dataset_type):
        super().__init__(data_root_path, augmentations, logger)
        self.dataset_type = dataset_type

        image_root = os.path.join(data_root_path, "training")
        image_dirs = {'left': [], 'right': []}
        disparity_dirs = {'left': []}

        for cam in ["left", "right"]:
            image_dirs[cam] = sorted(glob(os.path.join(image_root, f"{self.dataset_type}_{cam}/*")))
        disparity_dirs["left"] = [path.replace(f"{self.dataset_type}_left", "disparities") for path in image_dirs["left"]]

        num_seq = len(image_dirs["left"])
        for seq_idx in range(num_seq):
            sample = {
                'image': {'left': [], 'right': []},
                'disparity': {'left': []}
            }
            for cam in ["left", "right"]:
                sample["image"][cam] = sorted(glob(os.path.join(image_dirs[cam][seq_idx], "*.png")))
            sample["disparity"]["left"] = sorted(glob(os.path.join(disparity_dirs["left"][seq_idx], "*.png")))

            for img, disp in zip(sample["image"]["left"], sample["disparity"]["left"]):
                assert (img.split("/")[-1].split(".")[0] == disp.split("/")[-1].split(".")[0]), (img.split("/")[-1].split(".")[0], disp.split("/")[-1].split(".")[0])

            self.sample_list.append(sample)

    def __getitem__(self, index):
        sample = self.sample_list[index]
        sample_size = len(sample["image"]["left"])

        output = defaultdict(list)
        output_keys = ["img", "disp", "valid_disp"]
        for key in output_keys:
            output[key] = [[] for _ in range(sample_size)]

        for i in range(sample_size):
            for cam in ["left", "right"]:
                img = Image.open(sample["image"][cam][i])
                img = np.array(img).astype(np.uint8)
                img = img[..., :3]
                output["img"][i].append(img)

            disp, valid_disp = disparity_reader(sample["disparity"]["left"][i])
            disp = np.array(disp).astype(np.float32)  # [h, w]
            valid_disp = np.array(valid_disp).astype(np.float32)  # [h, w]
            output["disp"][i].append(-disp)
            output["valid_disp"][i].append(valid_disp)

        # output: {
        #     "img": [[left_img, right_img], [left_img, right_img], ......],
        #     "disp": [[left_disp], [left_disp], ......],
        #     "valid_disp": [[valid_disp], [valid_disp], ......],
        # }

        if self.augmentations is not None:
            for t in self.augmentations:
                output = t(output)

        for i in range(sample_size):
            output["disp"][i][0] = np.expand_dims(output["disp"][i][0], axis=0)  # [1, h, w]

        res = self.format_output(output)
        # {
        #     'img': [num_frames, 2(l&r), 3(c), h, w],
        #     'disp': [num_frames, 1(l), 1(c), h, w],
        #     'valid_disp': [num_frames, 1(l), h, w],
        #  }
        return res
