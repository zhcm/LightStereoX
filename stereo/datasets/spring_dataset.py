
import os
import numpy as np
import h5py
from PIL import Image
from .dataset_template import DatasetTemplate


class SpringDataset(DatasetTemplate):
    def __init__(self, data_root_path, split_file, augmentations, return_right_disp):
        super().__init__(data_root_path, split_file, augmentations)
        self.return_right_disp = return_right_disp

    def __getitem__(self, idx):
        item = self.data_list[idx]
        full_paths = [os.path.join(self.root, x) for x in item]
        left_path, right_path, disp_path = full_paths

        left_img = Image.open(left_path).convert('RGB')
        left_img = np.array(left_img, dtype=np.float32)

        right_img = Image.open(right_path).convert('RGB')
        right_img = np.array(right_img, dtype=np.float32)

        disp_img = self._readDsp5Disp(disp_path)
        disp_img = np.ascontiguousarray(disp_img, dtype=np.float32)[::2, ::2]

        sample = {
            'left': left_img,
            'right': right_img,
            'disp': disp_img,
        }

        if self.return_right_disp:
            disp_right_path = disp_path.replace('left', 'right')
            right_disp_img = self._readDsp5Disp(disp_right_path)
            right_disp_img = np.ascontiguousarray(right_disp_img, dtype=np.float32)[::2, ::2]
            sample['disp_right'] = right_disp_img

        if self.augmentations is not None:
            for t in self.augmentations:
                sample = t(sample)

        sample['index'] = idx
        sample['name'] = left_path

        return sample
    
    def _readDsp5Disp(self, filename):
        with h5py.File(filename, "r", locking=False) as f:
            if "disparity" not in f.keys():
                raise IOError(f"File {filename} does not have a 'disparity' key. Is this a valid dsp5 file?")
            return f["disparity"][()]
