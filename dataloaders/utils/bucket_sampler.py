import numpy as np
from random import shuffle
from torch.utils.data import Dataset, Sampler, DataLoader

class BucketSampler(Sampler):
    """
    A custom sampler for PyTorch DataLoader that creates batches with
    sequences of similar lengths..
    
    Args:
        lengths (list): A list of sequence lengths for each data point.
        bucket_boundaries (list): A list of bucket boundaries to group sequences.
        batch_size (int, optional): The size of the batch.
        
    Returns:
        An iterable list of batches where each batch contains indices of similar length sequences.
    """

    def __init__(self, lengths, bucket_boundaries, batch_size=4):
        self.lengths = lengths
        self.ind_n_len = [(i, l) for i, l in enumerate(lengths)]
        self.bucket_boundaries = bucket_boundaries
        self.batch_size = max(batch_size, 2)
        
    def __iter__(self):
        data_buckets = self._create_data_buckets()
        iter_list = self._create_iter_list(data_buckets)
        shuffle(iter_list)

        for batch in iter_list:
            yield batch
    
    def __len__(self):
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size

    def _create_data_buckets(self):
        data_buckets = {bucket_id: [] for bucket_id in range(len(self.bucket_boundaries) + 1)}
        
        for index, seq_len in self.ind_n_len:
            bucket_id = self._get_bucket_id(seq_len)
            data_buckets[bucket_id].append(index)

        return data_buckets

    def _create_iter_list(self, data_buckets):
        iter_list = []
        for bucket in data_buckets.values():
            if len(bucket) == 0:
                continue

            shuffle(bucket)
            num_batches = (len(bucket) + self.batch_size - 1) // self.batch_size
            batch_size_adjusted = len(bucket) // num_batches
            remaining_sequences = len(bucket) % num_batches

            start_index = 0
            bucket_batches = []
            for i in range(num_batches):
                end_index = start_index + batch_size_adjusted + (1 if remaining_sequences > 0 else 0)
                bucket_batches.append(bucket[start_index:end_index])
                start_index = end_index
                remaining_sequences -= 1

            # Duplicate the single-sequence batch's index to form a full batch
            if len(bucket_batches[-1]) == 1:
                single_seq_batch = bucket_batches[-1]
                while len(single_seq_batch) < self.batch_size:
                    single_seq_batch.append(single_seq_batch[0])

            iter_list.extend(bucket_batches)

        return iter_list


    def _get_bucket_id(self, seq_length):
        buckets_min = [np.iinfo(np.int32).min] + self.bucket_boundaries
        buckets_max = self.bucket_boundaries + [np.iinfo(np.int32).max]
        conditions = np.logical_and(
            np.less_equal(buckets_min, seq_length),
            np.less(seq_length, buckets_max))
        
        bucket_id = np.min(np.where(conditions))
        return bucket_id
