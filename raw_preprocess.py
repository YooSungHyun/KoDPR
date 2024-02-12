import copy
import os
import json
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from rank_bm25 import BM25Okapi
from datasets import Dataset
import torch
import torch.multiprocessing as mp

tokenizer = AutoTokenizer.from_pretrained("team-lucid/deberta-v3-base-korean")
MODEL_MAX_LENGTH = 512
CHUNK_SIZE = 100


def raw_preprocess(templates_list, dir_path, output_list, error_list):
    for file_name in templates_list:

        with open(os.path.join(dir_path, file_name), "r", encoding="utf-8") as f:
            sample = json.load(f)
        
        for each in tqdm(sample["data"], desc=file_name):
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
                        print(
                            f"{qa['answers']['text']}, {tokenizer.decode(tokens)}\n\n{tokenizer.decode(temp_tokens)}"
                        )
                        error_list.append(copy.deepcopy(data_dict))
                    output_list.append(copy.deepcopy(data_dict))


def make_bm25_hard(train_list):
    datasets = Dataset.from_pandas(pd.DataFrame(data=train_list))
    tot_ctxs = datasets["context"]
    tokenized_corpus = [tokenizer.tokenize(doc) for doc in tot_ctxs]
    bm25 = BM25Okapi(tokenized_corpus)
    for i in tqdm(range(len(train_list)), desc="hard bm25 making"):
        query = train_list[i]["question"]

        tokenized_query = tokenizer.tokenize(query)
        bm25_100 = bm25.get_top_n(tokenized_query, tot_ctxs, n=100)

        train_list[i]["bm25_hard"] = dict()
        for source in bm25_100:
            if train_list[i]["bm25_hard"]:
                break
            source_datas = datasets.filter(lambda x: x["context"] == source)
            source_datas = source_datas.shuffle()
            for data in source_datas:
                if data["answer"] != train_list[i]["answer"]:
                    train_list[i]["bm25_hard"] = copy.deepcopy(data)
                    break


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
train_output = "train_data_bm25.json"
if not os.path.exists(os.path.join(train_dir, train_output)):
    train_list = []
    train_error_list = []
    file_name, exp = train_output.split(".")
    preproc_split = file_name.split("_")[:2]
    preproc_output = "_".join(preproc_split)+"_preproc."+exp
    if not os.path.exists(os.path.join(train_dir, preproc_output)):
        queue = mp.Queue()

        raw_preprocess(TRAIN_DATASETS_TEMPLATES_01, train_dir, train_list, train_error_list)
        with open(os.path.join(train_dir, preproc_output), "w", encoding="utf-8") as file:
            file.write(json.dumps(train_list, indent=4))
        with open(os.path.join(train_dir, f"{'_'.join(preproc_split)}_error.json"), "w", encoding="utf-8") as file:
            file.write(json.dumps(train_error_list, indent=4))
    else:
        with open(os.path.join(train_dir, preproc_output), "r", encoding="utf-8") as f:
            train_list = json.load(f)

    make_bm25_hard(train_list)
    with open(os.path.join(train_dir, train_output), "w", encoding="utf-8") as file:
        file.write(json.dumps(train_list, indent=4))

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
valid_output = "dev_data.json"
if not os.path.exists(os.path.join(valid_dir, valid_output)):
    valid_list = []
    valid_error_list = []

    raw_preprocess(VALID_DATASETS_TEMPLATES_01, valid_dir, valid_list, valid_error_list)
    with open(os.path.join(valid_dir, valid_output), "w", encoding="utf-8") as file:
        file.write(json.dumps(valid_list, indent=4))
    with open(os.path.join(valid_dir, "error_data.json"), "w", encoding="utf-8") as file:
        file.write(json.dumps(valid_error_list, indent=4))

TEST_DATASETS_TEMPLATES_01 = []
TEST_DATASETS_TEMPLATES_02 = []

print("@@@@@@@@@@ test raw preprocess @@@@@@@@@@")
test_dir = "./raw_data/test"
test_output = "test_data.json"
if not os.path.exists(os.path.join(test_dir, test_output)):
    test_list = []

    raw_preprocess(TEST_DATASETS_TEMPLATES_01, test_dir, test_list)
    with open(os.path.join(test_dir, test_output), "w", encoding="utf-8") as file:
        file.write(json.dumps(test_list, indent=4))
