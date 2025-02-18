# @Time    : 2023/11/9 16:50
# @Author  : zhangchenming
import os
import numpy as np
import torch.utils.data as torch_data


class SequenceDatasetTemplate(torch_data.Dataset):
    def __init__(self, data_root_path, augmentations, logger):
        super().__init__()
        self.data_root_path = data_root_path
        self.augmentations = augmentations
        self.logger = logger
        self.sample_list = []

    def __len__(self):
        return len(self.sample_list)

    @staticmethod
    def format_output(output):
        res = {}
        if 'pad' in output:
            pad_value = output.pop('pad')
            res['pad'] = pad_value

        for k, v in output.items():
            if k != "viewpoint" and k != "metadata":
                for i in range(len(v)):
                    if len(v[i]) > 0:
                        v[i] = np.stack(v[i])  # each frame stack
                if len(v) > 0 and (len(v[0]) > 0):
                    res[k] = np.stack(v)  # sequence stack

        return res
