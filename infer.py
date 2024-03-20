import logging
import os
from logging import StreamHandler

import deepspeed
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

    text_p = [
        """개인용자동차보험 특별약관 > 27) 품질인증부품 사용 특별약관

제2조 (보상하는 손해)

보험회사(이하 '회사'라 함)는 피보험자동차의 단독사고(가해자 불명사고를 포함함) 또는 일방과실사고로 보통약관 「자기차량손해」 또는 「자기차량손 해 단독사고 보상 특별약관」에 따라서 보험금이 지급되는 경우 이 특별약관 에서 정한 품질인증부품(*1)을 사용하여 수리한 때 OEM 부품 공시가격(*2)의 25%를 피보험자에게 지급하여 드립니다.
(*1) 이 특별약관에서 품질인증부품이라 함은 「자동차관리법」 제 30조의5에 따라 인증된 부품을 말합니다.
(*2) 이 특별약관에서 OEM(Original Equipment Manufactu- ring) 부품이라 함은 자동차 제조사에서 출고된 자동차에 장
착된 부품을 말하며, OEM 부품 공시가격이라 함은 자동차관 리법 제30조의 5 제③항에서 말하는 대체부품인증기관이 공 시하는 가격을 말합니다.""",
        """개인용자동차보험 보통약관 > 제4편 일반사항 > 제1장 보험계약의 성립

제38조 (보험계약의 성립)

① 이 보험계약은 보험계약자가 청약을 하고 보험회사가 승낙을 하면 성립 합니다. ② 보험계약자가 청약을 할 때 '제1회 보험료(보험료를 분납하기로 약정한 경우)' 또는 '보험료 전액(보험료를 일시에 지급하기로 약정한 경우)'(이 하 '제1회 보험료 등'이라 함)을 지급하였을 때, 보험회사가 이를 받은 날 부터 15일 이내에 승낙 또는 거절의 통지를 발송하지 않으면 승낙한 것 으로 봅니다. ③ 보험회사가 청약을 승낙했을 때에는 지체 없이 보험증권을 보험계약자에 게 드립니다. 그러나 보험계약자가 제1회 보험료 등을 지급하지 않은 경 우에는 그러하지 않습니다. ④ 보험계약이 성립되면 보험회사는 제42조(보험기간)의 규정에 따라 보험 기간의 첫 날부터 보상책임을 집니다. 다만, 보험계약자로부터 제1회 보 험료 등을 받은 경우에는, 그 이후 승낙 전에 발생한 사고에 대해서도 청 약을 거절할 사유가 없는 한 보상합니다.""",
        """개인용자동차보험 특별약관 > 21) 보험료 자동납입 특별약관

제2조 (보험료의 자동납입)

① 보험계약자는 보험료를 분할납입할 때는 이 특별약관에 따라 보험증권에 기재된 횟수 및 금액으로 자동이체 분할납입합니다. 그러나 「대인배상 Ⅰ」 및 「대물배상」은 이 특별약관이 적용되지 않으므로 보험료를 분할납 입할 수 없습니다. ② 보험료의 자동이체 납입일은 보험증권에 기재된 이체일자(이하 '약정이
체일'이라 함)로 합니다. ③ 위 제2항의 약정이체일은 보험청약서상에 열거된 이체가능일자 중에서 보험료 납입기일 이후에 최초로 다가오는 이체일자를 말합니다. 다만, 초 회보험료를 자동이체 납입할 경우의 초회보험료 약정이체일은 회사와 보 험계약자가 별도로 약정한 책임개시일자 이전의 이체일자를 말합니다. ④ 지정은행계좌의 이체가능 금액이 회사가 청구한 보험료에 미달할 경우에
는 보험료가 이체 납입되지 않습니다.""",
    ]
    text_q = [
        "OEM 부품 공시가격이란 무엇인가요?",
        "보험회사가 청약을 승낙했을 때 어떤 절차를 밟나요?",
        "어떤 보험에 대해서는 보험료를 분할납입할 수 없나요?",
    ]
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
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    args = dataclass_to_namespace(args, "inference_args")

    main(args)
