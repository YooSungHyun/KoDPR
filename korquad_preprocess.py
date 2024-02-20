import copy
import os
import json
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset
import pickle
import re
from utils.comfy import get_passage_file
from rank_bm25 import BM25Okapi
from datasets import Dataset
import multiprocessing as mp
import random
from collections import deque

tokenizer = AutoTokenizer.from_pretrained("team-lucid/deberta-v3-base-korean")
MODEL_MAX_LENGTH = 512
CHUNK_SIZE = 100


def _get_cand_ids(title, title_passage_map):
    """미리 구축한 ko-wiki 데이터에서 해당 title에 맞는 id들을 가지고 옵니다."""
    refined_title = None
    ret = title_passage_map.get(title, None)
    if not ret:
        refined_title = re.sub(r"\(.*\)", "", title).strip()
        ret = title_passage_map.get(refined_title, None)
    return ret, refined_title


def raw_preprocess(start_index, end_index, data, title_passage_map, output_list):
    data_tuples = []
    for idx in tqdm(range(start_index, end_index), desc="raw preprocess"):
        item = data[idx]
        title = item["title"].replace("_", " ")  # _를 공백문자로 변경
        para = item["paragraphs"]
        cand_ids, refined_title = _get_cand_ids(title, title_passage_map)
        # title이 없으면 재낀다.
        if cand_ids is None:
            continue
        # 가지고 있는 파일 중 해당 id가 들어있는 파일을 찾는다.
        target_file_p = get_passage_file(cand_ids)
        if target_file_p is None:
            continue
        with open(target_file_p, "rb") as f:
            target_file = pickle.load(f)
        # 위키에 있는 실제 문장단위 텍스트가 들어감
        contexts = {cand_id: target_file[cand_id].split("[SEP]")[1].strip() for cand_id in cand_ids}

        # 해당 id의 잘린 context를 들고옴
        for p in para:
            qas = p["qas"]
            for qa in qas:
                answer = qa["answers"][0]["text"]  # 아무 정답이나 뽑습니다.
                answer_pos = qa["answers"][0]["answer_start"]
                answer_clue_start = max(0, answer_pos - 5)
                answer_clue_end = min(len(p["context"]), answer_pos + len(answer) + 5)
                answer_clue = p["context"][
                    answer_clue_start:answer_clue_end
                ]  # gold passage를 찾기 위해서 +-5칸의 주변 text 활용
                question = qa["question"]
                # 정답이 5글자포함 완전 적중이면 완전 적중에 해당하는 패시지만 뽑고
                answer_p = [(p_id, c) for p_id, c in contexts.items() if answer_clue in c]
                # 완전적중이 없으면 answer text만 있으면 학습 데이터로 삼는다
                # 데이터를 대충이나마 늘리는 효과가 있을 듯도 하다.
                if not answer_p:
                    answer_p = [(p_id, c) for p_id, c in contexts.items() if answer in c]
                data_tuples.extend([(title, question, p_id, c, answer) for p_id, c in answer_p])
    tokenized_tuples = [
        {"title": t, "question": q, "doc_id": None, "id": id, "context": p, "answer": a}
        for t, q, id, p, a in tqdm(data_tuples, desc="tokenize")
    ]
    output_list.extend(tokenized_tuples)


def make_bm25_hard(start_index, end_index, tot_ctxs, bm25, datasets, train_list, output):
    context_to_indices = {}

    for index, item in enumerate(datasets):
        context_value = item["context"]
        if context_value in context_to_indices:
            context_to_indices[context_value].append(index)
        else:
            context_to_indices[context_value] = [index]

    for i in tqdm(range(start_index, end_index + 1), desc="hard bm25 making"):
        temp_context = copy.deepcopy(context_to_indices)
        query = train_list[i]["question"]

        tokenized_query = tokenizer.tokenize(query)
        bm25_100 = bm25.get_top_n(tokenized_query, tot_ctxs, n=100)
        q = deque(bm25_100)
        train_list[i]["bm25_hard"] = list()
        train_list_idx = 0
        while q and train_list_idx < 100:
            source = q[0]
            while temp_context[source]:
                idx = temp_context[source].pop()
                if datasets[idx]["answer"] != train_list[i]["answer"]:
                    # bm25에 대한 정답이 다른 모든 passage를 저장함.
                    train_list[i]["bm25_hard"].append(idx)
                    break
            if not temp_context[source]:
                q.popleft()
                temp_context.pop(source)
            if q and source == q[0]:
                # val이 source와 같으면 아직은 남아있다는 뜻 (뒤로 붙힌다.)
                q.rotate(-1)
            train_list_idx += 1

        if train_list[i]["bm25_hard"]:
            assert len(train_list[i]["bm25_hard"]) <= 100, "뭔가 이상한 놈이 추가된게 있다?"
            output.append(train_list[i])


def main():
    split = "train"
    print(f"@@@@@@@@@@ {split} raw preprocess @@@@@@@@@@")
    korquad_processed_path = f"./raw_data/{split}/korquad_1.0_{split}_processed.json"
    if not os.path.exists(korquad_processed_path):
        with open(f"./raw_data/{split}/KorQuAD_v1.0_{split}.json", "rt", encoding="utf8") as f:
            data = json.load(f)
        datasets = data["data"]
        num_processes = mp.cpu_count()
        num_processes = min(len(datasets), num_processes)
        with open("./raw_data/title_passage_map.p", "rb") as f:
            title_passage_map = pickle.load(f)

        train_list = mp.Manager().list()
        processes = []
        chunk_size = max(len(datasets) // num_processes, 1)
        for idx in range(num_processes):
            start_index = idx * chunk_size
            end_index = min((idx + 1) * chunk_size, len(datasets))
            p = mp.Process(
                target=raw_preprocess, args=(start_index, end_index, datasets, title_passage_map, train_list)
            )
            processes.append(p)
            p.start()

        for child in processes:
            child.join()

        train_list = list(train_list)

        with open(korquad_processed_path, "w", encoding="utf-8") as file:
            file.write(json.dumps(train_list, indent=4))
    else:
        with open(korquad_processed_path, "r", encoding="utf-8") as f:
            train_list = json.load(f)
        num_processes = mp.cpu_count()
        num_processes = min(len(train_list), num_processes)
        output = mp.Manager().list()
        processes = []
        chunk_size = max(len(train_list) // num_processes, 1)
        datasets = Dataset.from_pandas(pd.DataFrame(data=train_list))
        tot_ctxs = list(set(datasets["context"]))
        tokenized_corpus = [tokenizer.tokenize(doc) for doc in tot_ctxs]
        bm25 = BM25Okapi(tokenized_corpus)
        if split == "train":
            train_output = f"./raw_data/{split}/korquad_1.0_{split}_processed_bm25idx.json"
            with open(korquad_processed_path, "r", encoding="utf-8") as f:
                train_list = json.load(f)

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
            with open(train_output, "w", encoding="utf-8") as file:
                file.write(json.dumps(processed_data, indent=4))


if __name__ == "__main__":
    main()
