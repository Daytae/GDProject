from typing import Iterator, List

import numpy as np
import torch
from torch.utils.data import BatchSampler, Sampler, SequentialSampler


class SequenceLengthBatchSampler(BatchSampler):
    def __init__(self, peptides: List[str], bucket_size: int, batch_size: int, sampler: Sampler = None):
        """
        Bucketing sampler for batching sequences of similar length together.

        `peptides`: List of peptide strings
        `bucket_size`: Bucket size (i.e. if bucket_size=5, then sequences of length 10-15 will be in the same bucket)
        `batch_size`: Batch size
        `sampler`: Sampler to use for sampling indices. Needed because PyTorch tries to inject a DistributedSampler in a DDP context.
        """
        if sampler is None:
            sampler = SequentialSampler(peptides)

        super().__init__(sampler, batch_size, False)

        lengths = [len(p) for p in peptides]  # Simple length for peptides
        min_len = min(lengths)
        max_len = max(lengths)

        buckets_min = np.arange(min_len, max_len + bucket_size, bucket_size)

        self.buckets = {
            idx: [] for idx in buckets_min[1:]
        }
        for i, length in enumerate(lengths):
            bucket_id = max(np.digitize(length, buckets_min, right=True), 1)
            self.buckets[buckets_min[bucket_id]].append(i)

        # Convert buckets to tensors
        for k,v in self.buckets.items():
            self.buckets[k] = torch.tensor(v, dtype=torch.long)

    def __iter__(self) -> Iterator[List[int]]:
        samples = set(iter(self.sampler))

        seed = int(torch.empty((), dtype=torch.int64).random_().item())
        generator = torch.Generator()
        generator.manual_seed(seed)

        flattened = []
        for bucket in self.buckets.values():
            indices = bucket[torch.randperm(len(bucket), generator=generator)].tolist()
            flattened.extend([x for x in indices if x in samples])

        batches = []
        for i in range(0, len(flattened), self.batch_size):
            batches.append(flattened[i:i+self.batch_size])

        # Shuffle batch order
        shuffled = [
            batches[i] for i in torch.randperm(len(batches), generator=generator)
        ]
        return iter(shuffled)