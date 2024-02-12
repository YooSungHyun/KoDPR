import math
from typing import Any, Dict, Iterator, List, Optional

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers.tokenization_utils_base import BatchEncoding
from transformers.utils import logging
import random
from collections import deque

logger = logging.get_logger(__name__)


class DistributedUniqueSampler(DistributedSampler):
    r"""
    unique data batch made
    """

    """DistributedSampler를 상속받아 'answer' 값 중복 방지 로직을 추가한 사용자 정의 Sampler"""

    def __init__(self, dataset, batch_size, num_replicas=None, rank=None, shuffle=True, seed=42):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.batch_size = batch_size
        self.length = len(self.dataset)
        random.seed(seed)
        self.dataset.shuffle()
        # 카테고리별, indices 배열 생성
        self.answer_to_indices = {}
        for idx in range(self.length):
            answer = self.dataset[idx]["answer"]
            if answer not in self.answer_to_indices:
                self.answer_to_indices[answer] = []
            self.answer_to_indices[answer].append(idx)
        assert (
            len(self.answer_to_indices.keys()) >= self.batch_size
        ), "데이터의 카테고리가 batch_size보다 작습니다. negative sampling 학습이 불가합니다."
        self.indices = list()
        self.sampler_size = self.length
        self._make_batches()

    def _make_batches(self):
        # 중복 없이 배치 생성
        answer_keys = list(self.answer_to_indices.keys())
        random.shuffle(answer_keys)
        q = deque(answer_keys)
        # dict의 카테고리가 num_replicas * batch_size보다 작으면 무조건 중복이 발생할 수 밖에 없다. (4*4인데 5,1,1,1,1,1,1,1,1,1,1,1 인경우, 현재 로직상 맨 마지막에 5-4=1이 겹침)
        while len(q) >= self.num_replicas * self.batch_size:
            # text answer를 하나 꺼내서
            answer_key = q.popleft()
            if len(self.answer_to_indices[answer_key]) <= self.num_replicas:
                # num_replicas보다 작으면, 그냥 extend해도 해당 카테고리는 배치에 들어갈 일이 없다.
                self.indices.extend(self.answer_to_indices[answer_key])
                # data를 다 뺐으니, key 삭제
                self.answer_to_indices.pop(answer_key)
            else:
                # num_replicas보다 크면, gpu 4갠데, 카테고리 1개가 5개인 경우, 0,1,2,3,0 과 같이 무지성으로 붙히면 gpu 0번에서 중복데이터가 발생한다.
                for _ in range(self.num_replicas):
                    self.indices.append(self.answer_to_indices[answer_key].pop())
                # 이 경우에는 queue의 맨 끝에 붙혀서, 추후에 볼 수 있도록 만들어본다.
                q.append(answer_key)

        # 분산 환경에 맞게 인덱스 조정
        self.sampler_size = len(self.indices)
        self.indices = self.indices[self.rank : self.sampler_size : self.num_replicas]

    def __iter__(self):
        return iter(self.indices)
