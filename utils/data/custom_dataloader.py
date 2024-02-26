import torch
import math


class CustomDataLoader(torch.utils.data.DataLoader):
    def __init__(self, tokenizer, *args, **kwargs):
        """
        Creates a data loader for AudioDatasets.
        """
        super(CustomDataLoader, self).__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.collate_fn = self._collate_fn

    def _collate_fn(self, batch):
        p_input_ids = [{"input_ids": v["batch_p_input_ids"]} for v in batch]
        q_input_ids = [{"input_ids": v["batch_q_input_ids"]} for v in batch]

        batch_p = self.tokenizer.pad(p_input_ids, return_tensors="pt")
        batch_q = self.tokenizer.pad(q_input_ids, return_tensors="pt")

        return batch_p, batch_q

    def __len__(self):
        length = len(self.sampler) // self.batch_size
        return length
