import json
import multiprocessing as mp
import os
from transformers import AutoTokenizer
from utils.data.custom_sampler import DistributedUniqueBM25Sampler
from utils.data.jsonl_dataset import JsonlDataset
from collections import Counter
from tqdm import tqdm

if __name__ == "__main__":

    BATCH_SIZE = 128
    EPOCH = 200
    NUM_REPLICAS = 4
    tokenizer = AutoTokenizer.from_pretrained("team-lucid/deberta-v3-base-korean")
    output_name = "./raw_data/train/korquad_klue_bm25_sampler_indices.json"
    dataset_name = "./raw_data/train/korquad_klue_bm25idx.json"

    def preprocess(examples):
        # deberta는 cls, sep(eos) 자동으로 넣어줌
        batch_p = tokenizer(
            tokenizer.cls_token + examples["context"],
            add_special_tokens=False,
            return_token_type_ids=False,
            return_attention_mask=False,
        )
        batch_q = tokenizer(
            tokenizer.cls_token + examples["question"],
            add_special_tokens=False,
            return_token_type_ids=False,
            return_attention_mask=False,
        )
        examples["batch_p_input_ids"] = batch_p["input_ids"]
        examples["batch_q_input_ids"] = batch_q["input_ids"]
        return examples

    train_dataset = JsonlDataset(dataset_name, transform=preprocess)
    if not os.path.isfile(output_name):
        custom_train_sampler = DistributedUniqueBM25Sampler(
            dataset=train_dataset, batch_size=BATCH_SIZE, tokenizer=tokenizer, num_replicas=4, rank=0, seed=42
        )
        train_list = mp.Manager().list()
        num_processes = mp.cpu_count()
        divisors_of_b = [i for i in range(1, EPOCH + 1) if EPOCH % i == 0]
        max_divisor_under_a = max([divisor for divisor in divisors_of_b if divisor <= num_processes])
        num_processes = min(max_divisor_under_a, num_processes)
        processes = []
        chunk_size = max(EPOCH // num_processes, 1)
        for idx in range(num_processes):
            start_index = idx * chunk_size
            end_index = min((idx + 1) * chunk_size, EPOCH)
            p = mp.Process(target=custom_train_sampler.__make_indices__, args=(start_index, end_index, train_list))
            processes.append(p)
            p.start()

        for child in processes:
            child.join()

        train_list = list(train_list)
        with open(output_name, "w", encoding="utf-8") as file:
            file.write(json.dumps(train_list, indent=4))

    # with open(output_name, "r", encoding="utf-8") as f:
    #     train_list = json.load(f)

    # if dataset_name.index("bm25") > -1:
    #     batch_size = BATCH_SIZE * 2
    # else:
    #     batch_size = BATCH_SIZE

    # assert len(train_list) == EPOCH, "멀티프로세스 처리 뭔가 잘못된듯!"

    # for epoch in train_list:
    #     for rank in range(NUM_REPLICAS):
    #         indices = epoch["indices"][rank::NUM_REPLICAS]
    #         for i in tqdm(range(0, len(indices), batch_size), desc=str(rank)):
    #             cnt = Counter(train_dataset[indices[i : i + batch_size]]["answer"])
    #             assert cnt.most_common(1)[0][1] == 1, "answer가 중복되는 배치 발생!"
