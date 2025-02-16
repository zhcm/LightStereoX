# @Time    : 2023/11/9 16:50
# @Author  : zhangchenming
import os
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
