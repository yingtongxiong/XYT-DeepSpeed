from typing import TypeVar

import numpy as np
import torch

T_co = TypeVar("T_co", covariant=True)


def packed_collate_fn(batch, packed_length):

    """
    Collate function for packed input sequences.

    Args:
        batch (List[Dict]): List of dictionaries representing each sample in batch.
            Each dictionary contains "tokens", "labels", "type_ids", "cu_seqlens", and "indexes" keys.
        packed_length (int): The length of packed sequence.

    Returns:
        Tuple[Dict[str, torch.Tensor], torch.Tensor]: A tuple containing a dictionary of tensors with "input_ids",
            "cu_seqlens", "indexes", and "type_ids" keys, and the tensor of padded "labels".

    Raises:
        AssertionError: If the length of a sample is not equal to packed_length.
        AssertionError: If the shape of the padded "input_ids" tensor does not have the correct shape.
    """
    have_image = False
    xs, ys, cu_seqlens, indexes, ts, images = [], [], [], [], [], []
    for b in batch:
        assert (
            len(b["tokens"]) == packed_length
        ), f"length of a sample should be equal to packed_length, but got {len(b['tokens'])} and {packed_length})"
        assert (
            len(b["labels"]) == packed_length
        ), f"length of a sample should be equal to packed_length, but got {len(b['labels'])} and {packed_length})"
        assert (
            len(b["type_ids"]) == packed_length
        ), f"length of a sample should be equal to packed_length, but got {len(b['type_ids'])} and {packed_length})"

        tokens = [abs(w) for w in b["tokens"]]
        labels = [w if w > 0 else -100 for w in b["labels"]]

        if b.get("images", None) is not None:
            have_image = True
            cur_images = torch.stack(b["images"])
            images.append(cur_images)

        xs.append(torch.LongTensor(tokens))
        # The labels have been shifted here, so they are aligned with the output corresponding to the token
        ys.append(torch.LongTensor(labels))
        ts.append(torch.LongTensor(b["type_ids"]))
        cu_seqlens.append(torch.IntTensor(b["cu_seqlens"]))
        indexes.append(torch.LongTensor(b["indexes"]))

    xs = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True)
    ys = torch.nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=-100)
    ts = torch.nn.utils.rnn.pad_sequence(ts, batch_first=True, padding_value=0)
    indexes = torch.stack(indexes, dim=0)
    if len(set(map(len, cu_seqlens))) == 1:  # if has uniform length, then stack to save device transfer time
        cu_seqlens = torch.stack(cu_seqlens, dim=0)

    assert xs.shape[1] == packed_length, (xs.shape[1], packed_length)
    if have_image:
        return {"input_ids": xs, "cu_seqlens": cu_seqlens, "indexes": indexes, "type_ids": ts, "images": images}, ys
    else:
        return {"input_ids": xs, "cu_seqlens": cu_seqlens, "indexes": indexes, "type_ids": ts}, ys
    
    

class StaticBatchSampler:
    """
    A static batch sampler that generates batches with a fixed micro-batch size.

    Args:
        num_samples (int): The total number of samples in the dataset.
        batch_size (int): The batch size for the current rank. Defaults to 192.
        rampup_batch_size (str): A string with three space-separated integers representing the
                                 starting batch size, the increment, and the number of steps between
                                 each increment. For example, "192 24 8" means that the batch size
                                 starts at 192 and increases by 24 every 8 steps. Defaults to
                                 "6 2 8", which corresponds to a batch size of 2 for the first 6 steps.
        micro_bsz (int): The micro-batch size. Defaults to 2.
        seed (int): The random seed for shuffling the indices. Defaults to 0.
        drop_last (bool): If True, drop the last incomplete batch. Currently only supports True. Defaults to True.
        data_rank (int): The rank of the current process in the data parallel group. Defaults to 0.
        data_world_size (int): The number of processes in the data parallel group. Defaults to 1.
    """

    def __init__(
        self,
        datasets,
        batch_size=192,
        rampup_batch_size="6 2 8",
        micro_bsz=2,
        seed=0,
        drop_last=True,
        data_rank=0,
        data_world_size=1,
    ):
        assert drop_last is True, "Currently only support drop last"
        # if rampup_batch_size:
        #     # In the process increase to batch_size
        #     start_bsz, bsz_incre, incre_every = map(int, rampup_batch_size.split())
        # else:
            # start_bsz, bsz_incre, incre_every = batch_size, batch_size, 1
        start_bsz, bsz_incre, incre_every = batch_size, batch_size, 1
        self.raw_rampup_batch_size = rampup_batch_size
        self.start_bsz = start_bsz
        self.bsz_incre = bsz_incre
        self.incre_every = incre_every
        # if gpc.is_initialized(ParallelMode.PIPELINE):
        #     assert (
        #         batch_size - self.start_bsz
        #     ) % self.bsz_incre == 0, f"{batch_size} - {self.start_bsz} should be multiple of {self.bsz_incre}"
        #     assert batch_size % micro_bsz == 0, f"batch_size({batch_size}) should be multiple of micro_bsz({micro_bsz})"
        #     assert (
        #         self.start_bsz % micro_bsz == 0
        #     ), f"start_bsz({self.start_bsz}) should be multiple of micro_bsz({micro_bsz})"
        #     assert (
        #         self.bsz_incre % micro_bsz == 0
        #     ), f"bsz_incre({self.bsz_incre}) should be multiple of micro_bsz({micro_bsz})"

        self.batch_size = batch_size
        self.epoch = 0
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.batch_count = 0
        self.micro_bsz = micro_bsz
        self.data_rank = data_rank
        self.data_world_size = data_world_size
        self.num_consumed_samples_in_epoch = 0
        self.datasets = datasets
        self.num_samples = sum([len(ds) for ds in datasets])

        self.get_indices()  # get data

    def get_indices(self, old_indices=None):
        if old_indices is not None:
            assert (
                len(old_indices) <= self.num_samples
            ), f"The checkpoint has {len(old_indices)} samples, \
while the new restart use less samples ({self.num_samples})"

        else:
            old_indices = np.array([])

        # indices includes len(old_indices) but not self.num_samples
        indices = np.arange(len(old_indices), self.num_samples)
        self.rng_state = self.rng.get_state()
        self.rng.shuffle(indices)
        # Need to consider drop_last
        ramp_steps = (self.batch_size - self.start_bsz) // self.bsz_incre
        if self.batch_count < ramp_steps * self.incre_every:
            rampup_samples = 0
            for i in range(ramp_steps):
                rampup_samples += (i * self.bsz_incre + self.start_bsz) * self.incre_every
            assert (
                rampup_samples * self.data_world_size <= self.num_samples
            ), f"Too much rampup samples: \
{rampup_samples*self.data_world_size} Vs. self.num_samples: {self.num_samples}"

            num_samples = (self.num_samples - rampup_samples * self.data_world_size) // (
                self.batch_size * self.data_world_size
            )
            num_samples = num_samples * self.batch_size * self.data_world_size + rampup_samples * self.data_world_size
        else:
            num_samples = self.num_samples // (self.batch_size * self.data_world_size)
            num_samples = num_samples * self.batch_size * self.data_world_size
        indices = np.concatenate([old_indices, indices]).astype(int)  # It needs to be spliced with the previous
        indices = indices[:num_samples]
        self.indices = indices
        assert len(self.indices) >= self.batch_size, "The number of samples should be larger than batch_size"
        self.num_consumed_samples_in_epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.rng = np.random.RandomState(self.seed + self.epoch)

    def __len__(self):
        ramp_steps = (self.batch_size - self.start_bsz) // self.bsz_incre
        if self.batch_count < ramp_steps * self.incre_every:
            rampup_samples = 0
            for i in range(ramp_steps):
                rampup_samples += (i * self.bsz_incre + self.start_bsz) * self.incre_every
            assert (
                rampup_samples * self.data_world_size <= self.num_samples
            ), f"Too much rampup samples: {rampup_samples*self.data_world_size} \
Vs. self.num_samples: {self.num_samples}"

            num_batches = (self.num_samples - rampup_samples * self.data_world_size) // self.batch_size
            num_batches = num_batches // self.data_world_size + self.incre_every * ramp_steps
        else:
            num_batches = self.num_samples // self.batch_size // self.data_world_size

        return num_batches

    def __iter__(self):
        indices = self.indices[self.data_rank :: self.data_world_size]
        while self.num_consumed_samples_in_epoch < len(indices):
            batch_rampup_idx = self.batch_count // self.incre_every
            cur_batch_size = batch_rampup_idx * self.bsz_incre + self.start_bsz
            cur_batch_size = min(cur_batch_size, self.batch_size)
            batch = indices[self.num_consumed_samples_in_epoch : self.num_consumed_samples_in_epoch + cur_batch_size]
            self.num_consumed_samples_in_epoch += len(batch)  # Consider multiple processes.
            self.batch_count += 1
            yield batch

        self.get_indices()  # get a new round

    def state_dict(self):
        states = {
            "batch_size": self.batch_size,
            "raw_rampup_batch_size": self.raw_rampup_batch_size,
            "rng_state": self.rng_state,
            "epoch": self.epoch,
            "seed": self.seed,
            "data_world_size": self.data_world_size,
            "num_consumed_samples_in_epoch": self.num_consumed_samples_in_epoch,
            "batch_count": self.batch_count,  # The batch_count here is due to the existence of multiple processes,
            # the batch may be oversent, and it needs to be overwritten by the external batch_count
            "indices": self.indices,  # The sequence used to breakpoint retraining is the same as before
        }

        return states

    def load_state_dict(self, states):
        for name in ("data_world_size", "raw_rampup_batch_size", "seed"):  # 'batch_size'
            assert states[name] == getattr(self, name), (name, states[name], getattr(self, name))  # should not change
        self.rng.set_state(states["rng_state"])
        self.get_indices(old_indices=None)  # Regenerate indices based on random state
        self.epoch = states["epoch"]
        self.batch_count = states["batch_count"]
        self.num_consumed_samples_in_epoch = states["num_consumed_samples_in_epoch"]

    def copy(self):
        copy_sampler = StaticBatchSampler(
            self.datasets,
            self.batch_size,
            self.raw_rampup_batch_size,
            self.micro_bsz,
            self.seed,
            drop_last=True,
            data_rank=self.data_rank,
            data_world_size=self.data_world_size,
        )

        copy_sampler.load_state_dict(self.state_dict())
        return copy_sampler
