import random
from utils.abc_dataset import CustomDataset
import json
from collections.abc import Callable


class JsonlDataset(CustomDataset):
    def __init__(self, path: str, transform: Callable = None):
        self.path = path
        with open(self.path, "r", encoding="utf-8") as f:
            self.jsonl = json.load(f)
        self.transform = transform

    def shuffle(self):
        random.shuffle(self.jsonl)

    def __len__(self):
        return len(self.jsonl)

    def __getitem__(self, idx):
        if isinstance(idx, (int, slice)):
            xs = self.jsonl[idx]
        elif isinstance(idx, (tuple, list)):
            xs = [self.jsonl[i] for i in idx]

        dict_of_lists = {}
        if isinstance(idx, (slice, tuple, list)):
            for i in range(len(xs)):
                if self.transform:
                    xs[i] = self.transform(xs[i])

                for key, value in xs[i].items():
                    if key in dict_of_lists:
                        dict_of_lists[key].append(value)
                    else:
                        dict_of_lists[key] = [value]
        else:
            if self.transform:
                xs = self.transform(xs)
            dict_of_lists.update(xs)
        return dict_of_lists
