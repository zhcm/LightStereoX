# @Time    : 2024/10/24 20:45
# @Author  : zhangchenming
import os
import gzip
import numpy as np
import torch

from PIL import Image
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional
from pytorch3d.implicitron.dataset.types import FrameAnnotation as ImplicitronFrameAnnotation, load_dataclass
from pytorch3d.renderer.cameras import PerspectiveCameras
from pytorch3d.utils import opencv_from_cameras_projection
from .dataset_template import SequenceDatasetTemplate


@dataclass
class DynamicReplicaFrameAnnotation(ImplicitronFrameAnnotation):
    """A dataclass used to load annotations from json."""
    camera_name: Optional[str] = None


class DynamicReplicaDataset(SequenceDatasetTemplate):
    def __init__(self, data_root_path, augmentations, logger, sample_len, split="train"):
        super().__init__(data_root_path, augmentations, logger)
        self.split = split
        self.sample_len = sample_len
        self.depth_eps = 1e-5
        only_first_n_samples = 1

        frame_annotations_file = f"frame_annotations_{split}.jgz"
        with gzip.open(os.path.join(data_root_path, split, frame_annotations_file), "rt", encoding="utf8") as zipfile:
            frame_annots_list = load_dataclass(zipfile, List[DynamicReplicaFrameAnnotation])

        seq_annot = defaultdict(lambda: defaultdict(list))  # {seq_name: {'left': [], 'right': []}}
        for frame_annot in frame_annots_list:
            seq_annot[frame_annot.sequence_name][frame_annot.camera_name].append(frame_annot)

        for seq_name in seq_annot.keys():
            filenames = defaultdict(lambda: defaultdict(list))
            for cam in ["left", "right"]:
                for framedata in seq_annot[seq_name][cam]:
                    im_path = os.path.join(data_root_path, split, framedata.image.path)
                    depth_path = os.path.join(data_root_path, split, framedata.depth.path)
                    mask_path = os.path.join(data_root_path, split, framedata.mask.path)

                    assert os.path.isfile(im_path), im_path
                    if self.split == 'train':
                        assert os.path.isfile(depth_path), depth_path
                    assert os.path.isfile(mask_path), mask_path

                    filenames["image"][cam].append(im_path)
                    if os.path.isfile(depth_path):
                        filenames["depth"][cam].append(depth_path)
                    filenames["mask"][cam].append(mask_path)

                    filenames["viewpoint"][cam].append(framedata.viewpoint)
                    filenames["metadata"][cam].append([framedata.sequence_name, framedata.image.size])

                    for k in filenames.keys():
                        assert (len(filenames[k][cam]) == len(filenames["image"][cam]) > 0), framedata.sequence_name

            seq_len = len(filenames["image"][cam])
            self.logger.info("seq_len {} {}".format(seq_name, seq_len))

            if split == "train":
                for ref_idx in range(0, seq_len - 5 * (self.sample_len - 1), 3):
                    step = 1 if self.sample_len == 1 else np.random.randint(1, 6)  # 序列采样的步长, 最大5，np.random.randint默认不包含最大值
                    sample = defaultdict(lambda: defaultdict(list))
                    for cam in ["left", "right"]:
                        for idx in range(ref_idx, ref_idx + step * self.sample_len, step):
                            for k in filenames.keys():
                                if "mask" not in k:
                                    sample[k][cam].append(filenames[k][cam][idx])
                    self.sample_list.append(sample)
            else:
                step = self.sample_len if self.sample_len > 0 else seq_len
                counter = 0
                for ref_idx in range(0, seq_len, step):
                    sample = defaultdict(lambda: defaultdict(list))
                    for cam in ["left", "right"]:
                        for idx in range(ref_idx, ref_idx + step):
                            for k in filenames.keys():
                                sample[k][cam].append(filenames[k][cam][idx])
                    self.sample_list.append(sample)
                    counter += 1
                    if 0 < only_first_n_samples <= counter:
                        break

        assert len(self.sample_list) > 0, "No samples found"
        self.logger.info(f"Added {len(self.sample_list)} from Dynamic Replica {split}")

    def __getitem__(self, index):
        sample = self.sample_list[index]  # train没有mask，val有mask
        sample_size = len(sample["image"]["left"])

        output = defaultdict(list)
        output_keys = ["img", "disp", "valid_disp", "mask", "viewpoint", "metadata"]
        for key in output_keys:
            output[key] = [[] for _ in range(sample_size)]

        viewpoint_left = self._get_pytorch3d_camera(
            sample["viewpoint"]["left"][0],
            sample["metadata"]["left"][0][1],
            scale=1.0,
        )
        viewpoint_right = self._get_pytorch3d_camera(
            sample["viewpoint"]["right"][0],
            sample["metadata"]["right"][0][1],
            scale=1.0,
        )
        depth2disp_scale = self.depth2disparity_scale(
            viewpoint_left,
            viewpoint_right,
            torch.Tensor(sample["metadata"]["left"][0][1])[None],
        )

        for i in range(sample_size):
            for cam in ["left", "right"]:
                if "mask" in sample:
                    mask = Image.open(sample["mask"][cam][i])
                    mask = np.array(mask) / 255.0
                    output["mask"][i].append(mask)

                viewpoint = self._get_pytorch3d_camera(
                    sample["viewpoint"][cam][i],
                    sample["metadata"][cam][i][1],
                    scale=1.0,
                )
                output["viewpoint"][i].append(viewpoint)

                metadata = sample["metadata"][cam][i]
                output["metadata"][i].append(metadata)

                img = Image.open(sample["image"][cam][i])
                img = np.array(img).astype(np.uint8)
                img = img[..., :3]
                output["img"][i].append(img)

                depth = self._load_16big_png_depth(sample["depth"][cam][i])  # [h,w]
                depth_mask = depth < self.depth_eps
                depth[depth_mask] = self.depth_eps
                disp = depth2disp_scale / depth
                disp[depth_mask] = 0
                valid_disp = (disp < 512) * (1 - depth_mask)

                disp = np.array(disp).astype(np.float32)
                disp = np.stack([-disp, np.zeros_like(disp)], axis=-1)
                output["disp"][i].append(disp)
                output["valid_disp"][i].append(valid_disp)

        if self.augmentations is not None:
            for t in self.augmentations:
                output = t(output)

        for i in range(sample_size):
            for cam in (0, 1):
                disp = output["disp"][i][cam][..., 0]
                valid_disp = (np.abs(disp) < 512) & (disp != 0)
                output["valid_disp"][i][cam] = valid_disp.astype(np.float32)
                output["disp"][i][cam] = np.expand_dims(disp, axis=0)
                if "mask" in output and cam < len(output["mask"][i]):
                    output["mask"][i][cam] = output["mask"][i][cam].astype(np.float32)

        res = self.format_output(output)
        # {
        #     'img': [num_frames, 2(l&r), 3(c), h, w],
        #     'disp': [num_frames, 2(l&r), 1(c), h, w],
        #     'valid_disp': [num_frames, 2(l&r), h, w],
        #     'mask': [num_frames, 2(l&r), h, w],
        #  }

        if self.split != "train":
            res["viewpoint"] = output["viewpoint"]
            res["metadata"] = output["metadata"]

        return res

    @staticmethod
    def _get_pytorch3d_camera(entry_viewpoint, image_size, scale: float) -> PerspectiveCameras:
        assert entry_viewpoint is not None
        # principal point and focal length
        principal_point = torch.tensor(entry_viewpoint.principal_point, dtype=torch.float)
        focal_length = torch.tensor(entry_viewpoint.focal_length, dtype=torch.float)

        half_image_size_wh_orig = (torch.tensor(list(reversed(image_size)), dtype=torch.float) / 2.0)

        # first, we convert from the dataset's NDC convention to pixels
        format = entry_viewpoint.intrinsics_format
        if format.lower() == "ndc_norm_image_bounds":
            # this is e.g. currently used in CO3D for storing intrinsics
            rescale = half_image_size_wh_orig
        elif format.lower() == "ndc_isotropic":
            rescale = half_image_size_wh_orig.min()
        else:
            raise ValueError(f"Unknown intrinsics format: {format}")

        # principal point and focal length in pixels
        principal_point_px = half_image_size_wh_orig - principal_point * rescale
        focal_length_px = focal_length * rescale

        # now, convert from pixels to PyTorch3D v0.5+ NDC convention
        # if self.image_height is None or self.image_width is None:
        out_size = list(reversed(image_size))

        half_image_size_output = torch.tensor(out_size, dtype=torch.float) / 2.0
        half_min_image_size_output = half_image_size_output.min()

        # rescaled principal point and focal length in ndc
        principal_point = (half_image_size_output - principal_point_px * scale) / half_min_image_size_output
        focal_length = focal_length_px * scale / half_min_image_size_output

        return PerspectiveCameras(
            focal_length=focal_length[None],
            principal_point=principal_point[None],
            R=torch.tensor(entry_viewpoint.R, dtype=torch.float)[None],
            T=torch.tensor(entry_viewpoint.T, dtype=torch.float)[None],
        )

    @staticmethod
    def depth2disparity_scale(left_camera, right_camera, image_size_tensor):
        # # opencv camera matrices
        (_, T1, K1), (_, T2, _) = [opencv_from_cameras_projection(f, image_size_tensor,) for f in (left_camera, right_camera)]
        fix_baseline = T1[0][0] - T2[0][0]
        focal_length_px = K1[0][0][0]
        # following this https://github.com/princeton-vl/RAFT-Stereo#converting-disparity-to-depth
        return focal_length_px * fix_baseline

    @staticmethod
    def _load_16big_png_depth(depth_png):
        with Image.open(depth_png) as depth_pil:
            # the image is stored with 16-bit depth but PIL reads it as I (32 bit).
            # we cast it to uint16, then reinterpret as float16, then cast to float32
            depth = (
                np.frombuffer(np.array(depth_pil, dtype=np.uint16), dtype=np.float16)
                .astype(np.float32)
                .reshape((depth_pil.size[1], depth_pil.size[0]))
            )
        return depth
