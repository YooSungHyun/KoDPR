import transformers
from transformers import DebertaV2Model
from transformers.models.deberta_v2.modeling_deberta_v2 import ContextPooler

import torch
import logging
import os
from copy import deepcopy


class KobertBiEncoder(torch.nn.Module):
    def __init__(self):
        super(KobertBiEncoder, self).__init__()
        self.passage_encoder = DebertaV2Model.from_pretrained("team-lucid/deberta-v3-base-korean")
        self.query_encoder = DebertaV2Model.from_pretrained("team-lucid/deberta-v3-base-korean")

    def forward(
        self, x: torch.LongTensor, attn_mask: torch.LongTensor, type: str = "passage"
    ) -> torch.FloatTensor:
        """passage 또는 query를 bert로 encoding합니다."""
        assert type in (
            "passage",
            "query",
        ), "type should be either 'passage' or 'query'"
        if type == "passage":
            last_hidden_states = self.passage_encoder(
                input_ids=x, attention_mask=attn_mask
            ).last_hidden_state
            average_pooling = torch.mean(last_hidden_states, dim=1)
            return average_pooling
        else:
            last_hidden_states = self.query_encoder(
                input_ids=x, attention_mask=attn_mask
            ).last_hidden_state
            average_pooling = torch.mean(last_hidden_states, dim=1)
            return average_pooling

    def checkpoint(self, model_ckpt_path):
        torch.save(deepcopy(self.state_dict()), model_ckpt_path)
        print(f"model self.state_dict saved to {model_ckpt_path}")

    def load(self, model_ckpt_path):
        with open(model_ckpt_path, "rb") as f:
            state_dict = torch.load(f)
        self.load_state_dict(state_dict)
        print(f"model self.state_dict loaded from {model_ckpt_path}")