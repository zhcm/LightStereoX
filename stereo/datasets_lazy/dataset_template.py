# @Time    : 2023/11/9 16:50
# @Author  : zhangchenming
import os
import torch.utils.data as torch_data


class DatasetTemplate(torch_data.Dataset):
    def __init__(self, data_root_path, split_file, augmentations):
        super().__init__()
        self.root = data_root_path
        self.split_file = split_file

        self.data_list = []
        if os.path.exists(self.split_file):
            with open(self.split_file, 'r') as fp:
                self.data_list.extend([x.strip().split(' ') for x in fp.readlines()])

        self.augmentations = augmentations

    def __len__(self):
        return len(self.data_list)
