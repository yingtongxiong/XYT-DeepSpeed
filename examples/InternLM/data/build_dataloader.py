from functools import partial

import torch.distributed as dist
from torch.utils.data import ConcatDataset, DataLoader

from .random_data import RandomDataset
from .packed_dataset import PackedDatasetWithoutCuSeqlen, PackedDatasetWithCut
from .sampler import StaticBatchSampler, packed_collate_fn

def get_train_dataloader(data_cfg):

    data_parallel_world_size = dist.get_world_size()
    train_ds = RandomDataset(
        num_samples=data_parallel_world_size * 500,
        max_len=data_cfg.seq_len,
        fixed_seqlen=data_cfg.fixed_random_dataset_seqlen,
    )
    
    if data_cfg.pack_sample_into_one:
        train_ds = PackedDatasetWithoutCuSeqlen(
            train_ds, max_length_per_sample=data_cfg.seq_len, packed_length=data_cfg.packed_length
        )
    else:
        train_ds = PackedDatasetWithCut(
            train_ds, max_length_per_sample=data_cfg.seq_len, packed_length=data_cfg.packed_length
        )

    train_sampler = StaticBatchSampler(
        train_ds.datasets if isinstance(train_ds, ConcatDataset) else [train_ds],
        batch_size=data_cfg.micro_num,
        rampup_batch_size=data_cfg.rampup_batch_size,
        micro_bsz=data_cfg.micro_bsz,
        seed=1024,
        drop_last=True,
        data_rank=dist.get_rank(),
        data_world_size=data_parallel_world_size,
    )
    train_collate_fn = partial(packed_collate_fn, packed_length=data_cfg.packed_length)
    
    # Create the training data loader
    train_dl = DataLoader(
        dataset=train_ds,
        batch_sampler=train_sampler,
        num_workers=data_cfg.num_worker,
        pin_memory=True,
        collate_fn=train_collate_fn,
        persistent_workers=data_cfg.num_worker > 0,
    )
    
    return train_dl