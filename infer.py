import logging
import os
from logging import StreamHandler

import torch
from arguments.inference_args import InferenceArguments
from networks.models import KobertBiEncoder
from setproctitle import setproctitle
from simple_parsing import ArgumentParser
from transformers import AutoTokenizer, DebertaV2Config, DebertaV2Model
from utils.comfy import dataclass_to_namespace, seed_everything
from utils.model_checkpointing.ds_handler import load_checkpoint_for_infer

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(levelname)8s] %(message)s")

timeFileHandler = StreamHandler()
timeFileHandler.setFormatter(formatter)

logger.addHandler(timeFileHandler)


def main(hparams: InferenceArguments):
    # reference: https://www.kaggle.com/code/anitarostami/lstm-multivariate-forecasting
    setproctitle("infer")
    seed_everything(hparams.seed)
    device = "cuda:0"
    tokenizer = AutoTokenizer.from_pretrained(hparams.transformers_model_name)

    text_p = []
    text_q = []
    assert len(text_q) == len(text_p), "입력값 에러!"
    for i in range(len(text_q)):
        text_q[i] = tokenizer.cls_token + text_q[i]
        text_p[i] = tokenizer.cls_token + text_p[i]

    # deberta는 cls, sep(eos) 자동으로 넣어줌
    batch_p = tokenizer(
        text_p, add_special_tokens=False, return_token_type_ids=False, padding=True, return_tensors="pt"
    ).to(device)
    batch_q = tokenizer(
        text_q, add_special_tokens=False, return_token_type_ids=False, padding=True, return_tensors="pt"
    ).to(device)

    p_config = DebertaV2Config.from_pretrained(hparams.model_path)
    q_config = DebertaV2Config.from_pretrained(hparams.model_path)
    p_encoder = DebertaV2Model(config=p_config)
    q_encoder = DebertaV2Model(config=q_config)

    # Instantiate objects
    model = KobertBiEncoder(passage_encoder=p_encoder, query_encoder=q_encoder, pooler=hparams.pooler).cuda(device)
    model.passage_encoder.eval()
    model.query_encoder.eval()
    torch.set_grad_enabled(False)
    state = {"model": model}
    load_checkpoint_for_infer(
        state,
        checkpoint_filepath=hparams.model_path,
        model_file_name="mp_rank_00_model_states.pt",
        device=device,
        logger=logger,
    )
    model = state["model"]

    p_emb = model(batch_p, "passage")  # bsz x bert_dim
    q_emb = model(batch_q, "query")  # bsz x bert_dim
    pred = torch.matmul(q_emb, p_emb.T)  # bsz x bsz
    print(pred)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(InferenceArguments, dest="inference_args")
    args = parser.parse_args()
    args = dataclass_to_namespace(args, "inference_args")

    main(args)
