# @Time    : 2023/11/9 16:50
# @Author  : zhangchenming
import os
import torch.utils.data as torch_data


class SequenceDatasetTemplate(torch_data.Dataset):
    def __init__(self, augmentations):
        super().__init__()
        self.sample_list = []
        self.augmentations = augmentations

    def __len__(self):
        return len(self.sample_list)
