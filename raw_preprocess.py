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
from collections import deque

tokenizer = AutoTokenizer.from_pretrained("team-lucid/deberta-v3-base-korean")
MODEL_MAX_LENGTH = 512
CHUNK_SIZE = 100


def raw_preprocess(start_index, end_index, data, output_list, error_list):
    for idx in tqdm(range(start_index, end_index), desc="raw preprocess"):
        each = data[idx]
        data_dict = dict()
        data_dict["doc_id"] = ""

        data_dict["title"] = ""
        for key_name in each.keys():
            if key_name in ["doc_id"]:
                data_dict["doc_id"] = each["doc_id"]
            if key_name in ["doc_title", "title"]:
                if data_dict["title"] == "":
                    data_dict["title"] = each[key_name]
                else:
                    data_dict["title"] = data_dict["title"] + "\n" + each[key_name]
        for paragraph in each["paragraphs"]:
            temp = list()
            for qa in paragraph["qas"]:
                answers_is_list = isinstance(qa["answers"], list)
                # 답변 가능한 질문 세트만 학데로 사용한다.
                if (
                    ("is_impossible" in qa.keys() and qa["is_impossible"])
                    or ("qa_type" in qa.keys() and qa["qa_type"] != 1)
                    or (answers_is_list and len(qa["answers"]) > 1)
                ):
                    continue

                # answers가 list[dict]인 경우와 dict인 경우를 통일해주기 위함
                if answers_is_list:
                    qa["answers"] = qa["answers"][0]

                # 답변이 겹치는 경우에는 질문을 사용하지 않는다.
                if not temp:
                    temp.append(qa["answers"])
                else:
                    if qa["answers"] in temp:
                        continue

                answer_start = qa["answers"]["answer_start"]
                answer_end = len(qa["answers"]["text"]) + qa["answers"]["answer_start"]
                if paragraph["context"][answer_start:answer_end] != qa["answers"]["text"]:
                    continue
                encoded = tokenizer(
                    paragraph["context"],
                    return_offsets_mapping=True,
                    add_special_tokens=False,
                )
                tokens = encoded.input_ids
                encoded_answer = tokenizer(qa["answers"]["text"], add_special_tokens=False).input_ids
                if len(encoded_answer) >= CHUNK_SIZE - 5:
                    print("답변이 너무 길면 사용되지 않습니다!")
                    continue
                offset_mappings = encoded.offset_mapping
                encoded_start = -1
                encoded_end = -1
                encoded_len = len(tokens)
                # korquad는 5글자 정답 완전 적중, 부분 적중으로 wiki passage로 갈아치워 데이터를 불릴겸 패시지를 정리한다.
                # 일반 한국어 데이터셋은 wiki와 같은 참고할 문헌이 없으므로, 최대한 답변이 상대적인 위치에 있을 수 있도록 불필요한 앞뒤
                # 문장을 잘라낸다.
                s = 0
                e = encoded_len
                temp_tokens = copy.deepcopy(tokens)
                if encoded_len > CHUNK_SIZE:
                    for i, (start, end) in enumerate(offset_mappings):
                        if encoded_start > -1 and encoded_end > -1:
                            break
                        if start <= answer_start <= end and encoded_start == -1:
                            encoded_start = i
                        if start <= answer_end <= end and encoded_end == -1:
                            encoded_end = i
                    # [(372, 376), (376, 377), (378, 379), (380, 383)] 149:153

                    while e - s > CHUNK_SIZE:
                        if encoded_start - s < e - encoded_end:
                            e -= 1
                        elif encoded_start - s > e - encoded_end:
                            s += 1
                        else:
                            e -= 1
                            s += 1
                    while "##" in tokenizer.decode(temp_tokens[s]):
                        # tokenizer가 단어단위로 안끝난경우 강제로 길이를 늘려서 단어단위로 끝내줌
                        s -= 1
                temp_tokens = temp_tokens[s:e]

                data_dict["question"] = qa["question"]
                data_dict["context"] = tokenizer.decode(temp_tokens)
                # data_dict["answer_start"] = qa["answers"]["answer_start"]
                data_dict["answer"] = qa["answers"]["text"]
                data_dict["id"] = None
                # 혹시몰라서 예외로 잡히는 부분은 따로 빼준다
                # 실제로 텍스트가 포함되어 있지 않거나, 토큰기준으로 포함되지 않은 경우에 기록된다.
                if data_dict["context"].replace(" ", "").find(
                    qa["answers"]["text"].replace(" ", "")
                ) == -1 and not all(item in temp_tokens for item in encoded_answer):
                    # 답변이 너무 긴 경우 다 잘려나간다.
                    print(f"{qa['answers']['text']}, {tokenizer.decode(tokens)}\n\n{tokenizer.decode(temp_tokens)}")
                    error_list.append(copy.deepcopy(data_dict))
                else:
                    output_list.append(copy.deepcopy(data_dict))


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
    TRAIN_DATASETS_TEMPLATES_01 = [
        "TL_span_extraction.json",
        "TL_span_inference.json",
        "TL_text_entailment.json",
        "행정문서1.json",
        "행정문서2.json",
        "행정문서3.json",
        "행정문서4.json",
        "행정문서5.json",
        "도서_220419_add.json",
        "ko_nia_normal_squad_all.json",
        "ko_nia_clue0529_squad_all.json",
    ]

    print("@@@@@@@@@@ train raw preprocess @@@@@@@@@@")
    train_dir = "./raw_data/train"
    train_output = "raw_preproc_bm25idx.json"
    if not os.path.exists(os.path.join(train_dir, train_output)):
        train_list = mp.Manager().list()
        train_error_list = mp.Manager().list()
        num_processes = mp.cpu_count()
        file_name, exp = train_output.split(".")
        preproc_split = file_name.split("_")[:2]
        preproc_output = "_".join(preproc_split) + "." + exp
        if not os.path.exists(os.path.join(train_dir, preproc_output)):
            data = list()

            for file_name in TRAIN_DATASETS_TEMPLATES_01:
                with open(os.path.join(train_dir, file_name), "r", encoding="utf-8") as f:
                    sample = json.load(f)
                data.extend(sample["data"])

            num_processes = min(len(data), num_processes)
            processes = []
            chunk_size = max(len(data) // num_processes, 1)
            for idx in range(num_processes):
                start_index = idx * chunk_size
                end_index = min((idx + 1) * chunk_size, len(data))
                p = mp.Process(
                    target=raw_preprocess, args=(start_index, end_index, data, train_list, train_error_list)
                )
                processes.append(p)
                p.start()

            for child in processes:
                child.join()

            train_list = list(train_list)
            error_list = list(train_error_list)

            with open(os.path.join(train_dir, preproc_output), "w", encoding="utf-8") as file:
                file.write(json.dumps(train_list, indent=4))
            with open(os.path.join(train_dir, f"{'_'.join(preproc_split)}_error.json"), "w", encoding="utf-8") as file:
                file.write(json.dumps(error_list, indent=4))
        else:
            with open(os.path.join(train_dir, preproc_output), "r", encoding="utf-8") as f:
                train_list = json.load(f)

            output = mp.Manager().list()
            processes = []
            chunk_size = max(len(train_list) // num_processes, 1)
            datasets = Dataset.from_pandas(pd.DataFrame(data=train_list))
            tot_ctxs = list(set(datasets["context"]))
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

    VALID_DATASETS_TEMPLATES_01 = [
        "VL_span_extraction.json",
        "VL_span_inference.json",
        "VL_text_entailment.json",
        "행정문서1.json",
        "행정문서2.json",
        "행정문서3.json",
        "행정문서4.json",
        "행정문서5.json",
        "도서.json",
    ]

    print("@@@@@@@@@@ dev raw preprocess @@@@@@@@@@")
    valid_dir = "./raw_data/dev"
    valid_output = "raw_preproc.json"
    if not os.path.exists(os.path.join(valid_dir, valid_output)):
        data = list()
        valid_list = []
        valid_error_list = []

        for file_name in VALID_DATASETS_TEMPLATES_01:
            with open(os.path.join(valid_dir, file_name), "r", encoding="utf-8") as f:
                sample = json.load(f)
            data.extend(sample["data"])

        raw_preprocess(0, len(data), data, valid_list, valid_error_list)
        with open(os.path.join(valid_dir, valid_output), "w", encoding="utf-8") as file:
            file.write(json.dumps(valid_list, indent=4))
        with open(os.path.join(valid_dir, "error_data.json"), "w", encoding="utf-8") as file:
            file.write(json.dumps(valid_error_list, indent=4))


if __name__ == "__main__":
    main()
