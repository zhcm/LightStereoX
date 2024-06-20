import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler


def build_dataloader(is_dist, all_dataset, batch_size, shuffle, workers, pin_memory):
    dataset = torch.utils.data.ConcatDataset(all_dataset)

    if is_dist:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
    else:
        sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    return dataset, loader, sampler
