import torch
from transformers import DebertaV2Model
from transformers.models.deberta_v2.modeling_deberta_v2 import ContextPooler


class KobertBiEncoder(torch.nn.Module):
    def __init__(self):
        super(KobertBiEncoder, self).__init__()
        self.passage_encoder = DebertaV2Model.from_pretrained("team-lucid/deberta-v3-base-korean")
        self.query_encoder = DebertaV2Model.from_pretrained("team-lucid/deberta-v3-base-korean")
        self.passage_encoder.gradient_checkpointing_enable()
        self.query_encoder.gradient_checkpointing_enable()

    def forward(self, batch, type: str = "passage") -> torch.FloatTensor:
        """passage 또는 query를 bert로 encoding합니다."""
        assert type in (
            "passage",
            "query",
        ), "type should be either 'passage' or 'query'"
        if type == "passage":
            last_hidden_states = self.passage_encoder(**batch).last_hidden_state
            average_pooling = torch.mean(last_hidden_states, dim=1)
            return average_pooling
        else:
            last_hidden_states = self.query_encoder(**batch).last_hidden_state
            average_pooling = torch.mean(last_hidden_states, dim=1)
            return average_pooling
