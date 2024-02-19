import copy
import os
import json
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from rank_bm25 import BM25Okapi
from datasets import Dataset
import multiprocessing as mp
import random

tokenizer = AutoTokenizer.from_pretrained("team-lucid/deberta-v3-base-korean")
MODEL_MAX_LENGTH = 512
CHUNK_SIZE = 100


def make_bm25_hard(start_index, end_index, tot_ctxs, bm25, datasets, train_list, output):
    context_to_indices = {}

    for index, item in enumerate(datasets):
        context_value = item["context"]
        if context_value in context_to_indices:
            context_to_indices[context_value].append(index)
        else:
            context_to_indices[context_value] = [index]

    for i in tqdm(range(start_index, end_index + 1), desc="hard bm25 making"):
        query = train_list[i]["question"]

        tokenized_query = tokenizer.tokenize(query)
        bm25_100 = bm25.get_top_n(tokenized_query, tot_ctxs, n=100)

        train_list[i]["bm25_hard"] = list()
        for source in bm25_100:
            source_indices = context_to_indices[source]
            for idx in source_indices:
                if datasets[idx]["answer"] != train_list[i]["answer"]:
                    # bm25에 대한 정답이 다른 모든 passage를 저장함.
                    train_list[i]["bm25_hard"].append(source)

        if train_list[i]["bm25_hard"]:
            output.append(train_list[i])


def main():
    print("@@@@@@@@@@ train raw preprocess @@@@@@@@@@")
    train_dir = "./raw_data/train"
    train_output = "total_preproc_bm25idx.json"
    if not os.path.exists(os.path.join(train_dir, train_output)):
        num_processes = mp.cpu_count()
        with open(os.path.join(train_dir, "total_preproc.json"), "r", encoding="utf-8") as f:
            train_list = json.load(f)

        random.shuffle(train_list)
        train_list = train_list[:150000]

        output = mp.Manager().list()
        processes = []
        chunk_size = max(len(train_list) // num_processes, 1)
        datasets = Dataset.from_pandas(pd.DataFrame(data=train_list))
        tot_ctxs = datasets["context"]
        tokenized_corpus = [tokenizer.tokenize(doc) for doc in tot_ctxs]
        bm25 = BM25Okapi(tokenized_corpus)
        for idx in range(num_processes):
            start_index = idx * chunk_size
            end_index = min((idx + 1) * chunk_size, len(train_list))
            p = mp.Process(
                target=make_bm25_hard, args=(start_index, end_index, tot_ctxs, bm25, datasets, train_list, output)
            )
            processes.append(p)
            p.start()
        for child in processes:
            child.join()

        processed_data = list(output)

        with open(os.path.join(train_dir, train_output), "w", encoding="utf-8") as file:
            file.write(json.dumps(processed_data, indent=4))


if __name__ == "__main__":
    main()
