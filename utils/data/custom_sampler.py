import copy
import random
import time
from collections import Counter, defaultdict, deque

from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers.utils import logging

logger = logging.get_logger(__name__)


class DistributedUniqueSampler(DistributedSampler):
    r"""
    unique data batch made
    DistributedSampler를 상속받아 'answer' 값 중복 방지 로직을 추가한 사용자 정의 Sampler
    """

    def __init__(self, dataset, batch_size, tokenizer, num_replicas=None, rank=None, shuffle=True, seed=42):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed)
        self.batch_size = batch_size
        self.length = len(self.dataset)
        self.tokenizer = tokenizer
        self.sampler_size = self.length
        self._init_data_structures()

    def _init_data_structures(self):
        random.seed(self.seed + self.epoch)
        # answer가 중복되질 않길 바라며 입력할 예정
        # 전처리로 저장된 bm25 passage text들에 대한 실제 dataset의 indices를 찾기 위함
        self.answer_to_indices = {}
        self.context_to_indices = {}
        for index, item in enumerate(self.dataset):
            answer = item["answer"]
            if answer in self.answer_to_indices:
                self.answer_to_indices[answer].append(index)
            else:
                self.answer_to_indices[answer] = [index]

        self.ori_answer_to_indices = copy.deepcopy(self.answer_to_indices)
        self.answer_set = set(self.ori_answer_to_indices.keys())
        assert (
            len(self.answer_to_indices.keys()) >= self.batch_size
        ), "데이터의 카테고리가 batch_size보다 작습니다. negative sampling 학습이 불가합니다."

    def __iter__(self):
        start = time.time()
        # 중복 없이 배치 생성
        self._init_data_structures()
        indices = []
        answer_keys = list(self.answer_to_indices.keys())
        # answer_keys를 섞기 때문에, 매 에포크별 최소한의 랜덤성이 부여된다.
        if self.shuffle:
            random.shuffle(answer_keys)

        q = deque(answer_keys)
        # dict의 카테고리가 num_replicas * batch_size보다 작으면 무조건 중복이 발생할 수 밖에 없다. (4*4인데 5,1,1,1,1,1,1,1,1,1,1,1 인경우, 현재 로직상 맨 마지막에 5-4=1이 겹침)
        while len(q) >= self.num_replicas * self.batch_size:
            # text answer를 하나 꺼내서
            answer_key = q.popleft()
            if len(self.answer_to_indices[answer_key]) <= self.num_replicas:
                # num_replicas보다 작으면, 그냥 extend해도 해당 카테고리는 배치에 중복으로 들어갈 일이 없다.
                for idx in self.answer_to_indices[answer_key]:
                    indices.append(idx)
                    if len(indices) % (self.batch_size * self.num_replicas) == 0:
                        q.append(answer_key)
                        break
            else:
                # num_replicas보다 크면, gpu 4갠데, 카테고리 1개가 5개인 경우, 0,1,2,3,0 과 같이 무지성으로 붙히면 gpu 0번에서 중복데이터가 발생한다.
                for _ in range(self.num_replicas):
                    indices.append(self.answer_to_indices[answer_key].pop())
                    if len(indices) % (self.batch_size * self.num_replicas) == 0:
                        break
                # 이 경우에는 queue의 맨 끝에 붙혀서, 추후에 볼 수 있도록 만들어본다.
                q.append(answer_key)

            if len(self.answer_to_indices[answer_key]) == 0:
                # data를 다 뺐으니, key 삭제
                self.answer_to_indices.pop(answer_key)

        # 분산 환경에 맞게 인덱스 조정
        self.sampler_size = len(indices)
        indices = indices[self.rank :: self.num_replicas]
        end = time.time()

        logger.info(f"@@@@@@ {self.rank}_sampler make iter time: {end - start:.5f} sec")
        return iter(indices)

    def __len__(self):
        return len(list(self.__iter__()))


class DistributedUniqueBM25Sampler(DistributedSampler):
    r"""
    DistributedUniqueBM25Sampler
    DistributedSampler를 상속받아 'answer' 값 중복 방지 + BM25 중복 방지 로직을 추가한 사용자 정의 Sampler
    """

    def __init__(self, dataset, batch_size, tokenizer, num_replicas=None, rank=None, shuffle=True, seed=42):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed)
        self.batch_size = batch_size
        self.length = len(self.dataset)
        self.tokenizer = tokenizer
        self.sampler_size = self.length
        self._init_data_structures()

    def _init_data_structures(self):
        random.seed(self.seed + self.epoch)
        # answer가 중복되질 않길 바라며 입력할 예정
        # 전처리로 저장된 bm25 passage text들에 대한 실제 dataset의 indices를 찾기 위함
        self.answer_to_indices = {}
        self.context_to_indices = {}
        for index, item in enumerate(self.dataset):
            answer = item["answer"]
            if answer in self.answer_to_indices:
                self.answer_to_indices[answer].append(index)
            else:
                self.answer_to_indices[answer] = [index]

            ctx = item["context"]
            if ctx in self.context_to_indices:
                self.context_to_indices[ctx].append(index)
            else:
                self.context_to_indices[ctx] = [index]
        self.ori_answer_to_indices = copy.deepcopy(self.answer_to_indices)
        self.answer_set = set(self.ori_answer_to_indices.keys())
        assert (
            len(self.answer_to_indices.keys()) >= self.batch_size
        ), "데이터의 카테고리가 batch_size보다 작습니다. negative sampling 학습이 불가합니다."

    def __iter__(self):
        start = time.time()
        # 중복 없이 배치 생성
        self._init_data_structures()
        indices = []
        answer_keys = list(self.answer_to_indices.keys())
        # answer_keys를 섞기 때문에, 매 에포크별 최소한의 랜덤성이 부여된다.
        if self.shuffle:
            random.shuffle(answer_keys)

        q = deque(answer_keys)
        # dict의 카테고리가 num_replicas * batch_size보다 작으면 무조건 중복이 발생할 수 밖에 없다. (4*4인데 5,1,1,1,1,1,1,1,1,1,1,1 인경우, 현재 로직상 맨 마지막에 5-4=1이 겹침)
        while len(q) >= self.num_replicas * self.batch_size:
            # text answer를 하나 꺼내서
            answer_key = q.popleft()
            if len(self.answer_to_indices[answer_key]) <= self.num_replicas:
                # num_replicas보다 작으면, 그냥 extend해도 해당 카테고리는 배치에 중복으로 들어갈 일이 없다.
                for idx in self.answer_to_indices[answer_key]:
                    indices.append(idx)
                    if len(indices) % (self.batch_size * self.num_replicas) == 0:
                        q.append(answer_key)
                        break
            else:
                # num_replicas보다 크면, gpu 4갠데, 카테고리 1개가 5개인 경우, 0,1,2,3,0 과 같이 무지성으로 붙히면 gpu 0번에서 중복데이터가 발생한다.
                for _ in range(self.num_replicas):
                    indices.append(self.answer_to_indices[answer_key].pop())
                    if len(indices) % (self.batch_size * self.num_replicas) == 0:
                        break
                # 이 경우에는 queue의 맨 끝에 붙혀서, 추후에 볼 수 있도록 만들어본다.
                q.append(answer_key)

            if len(self.answer_to_indices[answer_key]) == 0:
                # data를 다 뺐으니, key 삭제
                self.answer_to_indices.pop(answer_key)

            if len(indices) % (self.batch_size * self.num_replicas) == 0:
                # batch로 만들 것이 한바퀴 완성되었으면, 그 뒤에는 bm25 batch를 구성해줘야함
                # 마지막 batch_size의 num_replicas 만큼 잘라서,
                temp = copy.deepcopy(indices[-(self.batch_size * self.num_replicas) :])
                results = list()
                for replica in range(self.num_replicas):
                    each_replica = list()
                    answer_bm25_hist = list()
                    replicas_last_batch_indices = temp[replica :: self.num_replicas]
                    batch = self.dataset[replicas_last_batch_indices]
                    answer_bm25_hist.extend(batch["answer"])
                    batch_answer_cnt = Counter(batch["answer"])
                    bm25_visited = defaultdict(lambda: False)
                    assert batch_answer_cnt.most_common(1)[0][1] == 1, "batch내의 중복값 발생!"
                    for bm25_list in batch["bm25_hard"]:
                        # 각 batch의 각 데이터별 bm25_hard 리스트
                        for ctx in bm25_list:
                            # bm25_list는 ranking별로 내림차순 정렬되어있다.
                            try:
                                for idx in self.context_to_indices[ctx]:
                                    # batch 대상에 answer에 포함되지 않은 context 이면서,
                                    # bm25로도 방문하지 않은 answer인 경우
                                    # batch에도, bm25에도 포함되지 않는 데이터인 경우
                                    target = self.dataset[idx]["answer"]
                                    if batch_answer_cnt[target] == 0 and not bm25_visited[target]:
                                        each_replica.append(idx)
                                        answer_bm25_hist.append(target)
                                        bm25_visited[target] = True
                                        break
                            except KeyError:
                                # 전처리 진행 시 bm25에서는 검색이 되었으나, 정작 자기 스스로 qustion으로 bm25을 했을때
                                # 데이터가 없었던 경우, bm25_list에는 들어있지만, 실제 데이터에선 빠r질 여지가 있다.
                                continue
                            # 각 배치별로 1개의 bm25_hard를 추출하면 되므로, break로 다음 batch를 보도록 하자.
                            break
                    diff_cnt = len(replicas_last_batch_indices) - len(each_replica)
                    unique_answer = list(self.answer_set - set(answer_bm25_hist))
                    random.shuffle(unique_answer)
                    while diff_cnt > 0:
                        # 어떤 batch에 한해서는, bm25 결과가 없을 수도 있다.
                        # 논문에서는 이런 데이터는 제외하였다고 하지만, 생각해보면 contrastive learning에서
                        # 배치의 데이터 구조가 달라지면 결국 새로운 데이터라고 정의할 수 있다. 따라서 그냥 중복되지 않은 놈
                        # 아무거나 하나 넣는다.
                        # random으로 bm25랑은 관련 없지만, answer가 안겹치는 애들 삽입
                        value = unique_answer.pop()
                        if value in self.ori_answer_to_indices:
                            each_replica.append(self.ori_answer_to_indices[value][0])
                            diff_cnt -= 1

                    # 여기까지 당도하면, 무조건 bm25하나는 있는 데이터만 활용된다.
                    results.append(each_replica)

                for column in zip(*results):
                    indices.extend(column)
                    temp.extend(column)

                for replica in range(self.num_replicas):
                    replicas_last_batch_indices = temp[replica :: self.num_replicas]
                    batch = self.dataset[replicas_last_batch_indices]
                    batch_answer_cnt = Counter(batch["answer"])
                    assert batch_answer_cnt.most_common(1)[0][1] == 1, "bm25 완성 batch내의 중복값 발생!"
                    assert len(replicas_last_batch_indices) == self.batch_size * 2, "누락값 발생!"

        # 분산 환경에 맞게 인덱스 조정
        self.sampler_size = len(indices)
        indices = indices[self.rank :: self.num_replicas]
        end = time.time()

        logger.info(f"@@@@@@ {self.rank}_sampler make iter time: {end - start:.5f} sec")
        return iter(indices)

    def __len__(self):
        return len(list(self.__iter__()))

    def __make_indices__(self, start_index, end_index, outputs):
        # 중복 없이 배치 생성
        for process_idx in tqdm(range(start_index, end_index)):
            self.set_epoch(process_idx)
            self._init_data_structures()
            indices = []
            answer_keys = list(self.answer_to_indices.keys())
            # answer_keys를 섞기 때문에, 매 에포크별 최소한의 랜덤성이 부여된다.
            if self.shuffle:
                random.shuffle(answer_keys)

            q = deque(answer_keys)
            # dict의 카테고리가 num_replicas * batch_size보다 작으면 무조건 중복이 발생할 수 밖에 없다. (4*4인데 5,1,1,1,1,1,1,1,1,1,1,1 인경우, 현재 로직상 맨 마지막에 5-4=1이 겹침)
            while len(q) >= self.num_replicas * self.batch_size:
                # text answer를 하나 꺼내서
                answer_key = q.popleft()
                if len(self.answer_to_indices[answer_key]) <= self.num_replicas:
                    # num_replicas보다 작으면, 그냥 extend해도 해당 카테고리는 배치에 중복으로 들어갈 일이 없다.
                    for idx in self.answer_to_indices[answer_key]:
                        indices.append(idx)
                        if len(indices) % (self.batch_size * self.num_replicas) == 0:
                            q.append(answer_key)
                            break
                else:
                    # num_replicas보다 크면, gpu 4갠데, 카테고리 1개가 5개인 경우, 0,1,2,3,0 과 같이 무지성으로 붙히면 gpu 0번에서 중복데이터가 발생한다.
                    for _ in range(self.num_replicas):
                        indices.append(self.answer_to_indices[answer_key].pop())
                        if len(indices) % (self.batch_size * self.num_replicas) == 0:
                            break
                    # 이 경우에는 queue의 맨 끝에 붙혀서, 추후에 볼 수 있도록 만들어본다.
                    q.append(answer_key)

                if len(self.answer_to_indices[answer_key]) == 0:
                    # data를 다 뺐으니, key 삭제
                    self.answer_to_indices.pop(answer_key)

                if len(indices) % (self.batch_size * self.num_replicas) == 0:
                    # batch로 만들 것이 한바퀴 완성되었으면, 그 뒤에는 bm25 batch를 구성해줘야함
                    # 마지막 batch_size의 num_replicas 만큼 잘라서,
                    temp = copy.deepcopy(indices[-(self.batch_size * self.num_replicas) :])
                    results = list()
                    for replica in range(self.num_replicas):
                        each_replica = list()
                        answer_bm25_hist = list()
                        replicas_last_batch_indices = temp[replica :: self.num_replicas]
                        batch = self.dataset[replicas_last_batch_indices]
                        answer_bm25_hist.extend(batch["answer"])
                        batch_answer_cnt = Counter(batch["answer"])
                        bm25_visited = defaultdict(lambda: False)
                        assert batch_answer_cnt.most_common(1)[0][1] == 1, "batch내의 중복값 발생!"
                        for bm25_list in batch["bm25_hard"]:
                            # 각 batch의 각 데이터별 bm25_hard 리스트
                            for ctx in bm25_list:
                                # bm25_list는 ranking별로 내림차순 정렬되어있다.
                                try:
                                    for idx in self.context_to_indices[ctx]:
                                        # batch 대상에 answer에 포함되지 않은 context 이면서,
                                        # bm25로도 방문하지 않은 answer인 경우
                                        # batch에도, bm25에도 포함되지 않는 데이터인 경우
                                        target = self.dataset[idx]["answer"]
                                        if batch_answer_cnt[target] == 0 and not bm25_visited[target]:
                                            each_replica.append(idx)
                                            answer_bm25_hist.append(target)
                                            bm25_visited[target] = True
                                            break
                                except KeyError:
                                    # 전처리 진행 시 bm25에서는 검색이 되었으나, 정작 자기 스스로 qustion으로 bm25을 했을때
                                    # 데이터가 없었던 경우, bm25_list에는 들어있지만, 실제 데이터에선 빠r질 여지가 있다.
                                    continue
                                # 각 배치별로 1개의 bm25_hard를 추출하면 되므로, break로 다음 batch를 보도록 하자.
                                break
                        diff_cnt = len(replicas_last_batch_indices) - len(each_replica)
                        unique_answer = list(self.answer_set - set(answer_bm25_hist))
                        random.shuffle(unique_answer)
                        while diff_cnt > 0:
                            # 어떤 batch에 한해서는, bm25 결과가 없을 수도 있다.
                            # 논문에서는 이런 데이터는 제외하였다고 하지만, 생각해보면 contrastive learning에서
                            # 배치의 데이터 구조가 달라지면 결국 새로운 데이터라고 정의할 수 있다. 따라서 그냥 중복되지 않은 놈
                            # 아무거나 하나 넣는다.
                            # random으로 bm25랑은 관련 없지만, answer가 안겹치는 애들 삽입
                            value = unique_answer.pop()
                            if value in self.ori_answer_to_indices:
                                each_replica.append(self.ori_answer_to_indices[value][0])
                                diff_cnt -= 1

                        # 여기까지 당도하면, 무조건 bm25하나는 있는 데이터만 활용된다.
                        results.append(each_replica)

                    for column in zip(*results):
                        indices.extend(column)
                        temp.extend(column)

                    for replica in range(self.num_replicas):
                        replicas_last_batch_indices = temp[replica :: self.num_replicas]
                        batch = self.dataset[replicas_last_batch_indices]
                        batch_answer_cnt = Counter(batch["answer"])
                        assert batch_answer_cnt.most_common(1)[0][1] == 1, "bm25 완성 batch내의 중복값 발생!"
                        assert len(replicas_last_batch_indices) == self.batch_size * 2, "누락값 발생!"
            outputs.append({"epoch": self.epoch, "num_replicas": self.num_replicas, "indices": indices})
