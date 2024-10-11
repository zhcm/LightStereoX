import random
import torch

from functools import partial
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler


def build_dataloader(is_dist, all_dataset, batch_size, shuffle, workers, pin_memory, drop_last=False,
                     batch_uniform=False, h_range=None, w_range=None):
    dataset = torch.utils.data.ConcatDataset(all_dataset)

    if is_dist:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
    else:
        sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)

    partial_custom_collate = partial(custom_collate, concat_dataset=dataset,
                                     batch_uniform=batch_uniform, h_range=h_range, w_range=w_range)

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=workers,
        collate_fn=partial_custom_collate,
        pin_memory=pin_memory,
        drop_last=drop_last
    )
    return dataset, loader, sampler


def get_size(base_size, w_range, h_range):
    w = random.randint(int(w_range[0] * base_size[1]), int(w_range[1] * base_size[1]))
    h = random.randint(int(h_range[0] * base_size[0]), int(h_range[1] * base_size[0]))
    return int(h), int(w)


def custom_collate(data_list, concat_dataset, batch_uniform=False, h_range=None, w_range=None):
    if batch_uniform:
        for each_dataset in concat_dataset.datasets:
            for cur_t in each_dataset.augmentations:
                if type(cur_t).__name__ == 'RandomCrop':
                    base_size = cur_t.base_size
                    cur_t.crop_size = get_size(base_size, w_range, h_range)
                    break

    return torch.utils.data.default_collate(data_list)
