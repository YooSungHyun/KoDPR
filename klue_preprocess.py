import copy
import os
import json
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset
from rank_bm25 import BM25Okapi
from datasets import Dataset
import multiprocessing as mp

tokenizer = AutoTokenizer.from_pretrained("team-lucid/deberta-v3-base-korean")
klue = load_dataset("KLUE", name="mrc")
MODEL_MAX_LENGTH = 512
CHUNK_SIZE = 100


def raw_preprocess(start_index, end_index, data, output_list, error_list):
    for idx in tqdm(range(start_index, end_index), desc="raw preprocess"):
        each = data[idx]
        if each["is_impossible"]:
            continue
        data_dict = dict()
        data_dict["doc_id"] = each["guid"]
        data_dict["title"] = each["title"]
        context = each["context"]
        data_dict["question"] = each["question"]
        data_dict["id"] = None
        temp = list()
        for i in range(len(each["answers"]["answer_start"])):
            answer = each["answers"]["text"][i]
            answer_start = each["answers"]["answer_start"][i]
            answer_end = answer_start + len(answer)
            if not temp:
                temp.append(each["answers"]["answer_start"][i])
            else:
                if each["answers"]["answer_start"][i] in temp:
                    continue

            if context[answer_start:answer_end] != answer:
                print("context에 정답이 포함되어있지 않음", answer)
                print(context)
                continue

            encoded = tokenizer(context, return_offsets_mapping=True, add_special_tokens=False)
            tokens = encoded.input_ids
            encoded_answer = tokenizer(answer, add_special_tokens=False).input_ids
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
                for offset_idx, (start, end) in enumerate(offset_mappings):
                    if encoded_start > -1 and encoded_end > -1:
                        break
                    if start <= answer_start <= end and encoded_start == -1:
                        encoded_start = offset_idx
                    if start <= answer_end <= end and encoded_end == -1:
                        encoded_end = offset_idx
                # [(372, 376), (376, 377), (378, 379), (380, 383)] 149:153

                while e - s > CHUNK_SIZE:
                    start_bound = encoded_start - s
                    end_bound = e - encoded_end
                    if start_bound < end_bound:
                        e -= 1
                    elif start_bound > end_bound:
                        s += 1
                    else:
                        e -= 1
                        s += 1
                while "##" in tokenizer.decode(temp_tokens[s]):
                    # tokenizer가 단어단위로 안끝난경우 강제로 길이를 늘려서 단어단위로 끝내줌
                    s -= 1
            temp_tokens = temp_tokens[s:e]
            # 다 처리를 한 token에 unk가 있으면 학데에 쓰지 않는다.
            data_dict["answer"] = answer
            data_dict["context"] = tokenizer.decode(temp_tokens)
            # 혹시몰라서 예외로 잡히는 부분은 따로 빼준다
            # 실제로 텍스트가 포함되어 있지 않거나, 토큰기준으로 포함되지 않은 경우에 기록된다.
            if data_dict["context"].replace(" ", "").find(answer.replace(" ", "")) == -1 and not all(
                item in temp_tokens for item in encoded_answer
            ):
                print(f"{answer}, {tokenizer.decode(tokens)}\n\n{tokenizer.decode(temp_tokens)}")
                error_list.append(copy.deepcopy(data_dict))
            else:
                output_list.append(copy.deepcopy(data_dict))


def make_bm25_hard(start_index, end_index, tot_ctxs, bm25, datasets, train_list, output):
    for i in tqdm(range(start_index, end_index), desc="hard bm25 making"):
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
        if train_list[i]["bm25_hard"]:
            output.append(train_list[i])


def main():
    print("@@@@@@@@@@ train raw preprocess @@@@@@@@@@")
    train_dir = "./raw_data/train"
    train_output = "klue_data_bm25.json"
    if not os.path.exists(os.path.join(train_dir, train_output)):
        train_list = mp.Manager().list()
        train_error_list = mp.Manager().list()
        num_processes = mp.cpu_count()
        file_name, exp = train_output.split(".")
        preproc_split = file_name.split("_")[:2]
        preproc_output = "_".join(preproc_split) + "_preproc." + exp
        if not os.path.exists(os.path.join(train_dir, preproc_output)):
            data = klue["train"].to_list()
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

    print("@@@@@@@@@@ dev raw preprocess @@@@@@@@@@")
    valid_dir = "./raw_data/dev"
    valid_output = "klue_data.json"
    if not os.path.exists(os.path.join(valid_dir, valid_output)):
        valid_list = []
        valid_error_list = []

        raw_preprocess(
            0, len(klue["validation"].to_list()), klue["validation"].to_list(), valid_list, valid_error_list
        )
        with open(os.path.join(valid_dir, valid_output), "w", encoding="utf-8") as file:
            file.write(json.dumps(valid_list, indent=4))
        with open(os.path.join(valid_dir, "error_data.json"), "w", encoding="utf-8") as file:
            file.write(json.dumps(valid_error_list, indent=4))


if __name__ == "__main__":
    main()
