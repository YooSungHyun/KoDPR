import torch
from transformers.models.deberta_v2.modeling_deberta_v2 import ContextPooler


class KobertBiEncoder(torch.nn.Module):
    def __init__(self, passage_encoder, query_encoder, pooler=False):
        super(KobertBiEncoder, self).__init__()
        self.passage_encoder = passage_encoder
        self.passage_encoder.gradient_checkpointing_enable()
        self.query_encoder = query_encoder
        self.query_encoder.gradient_checkpointing_enable()
        self.passage_pooler = None
        self.query_pooler = None
        if pooler:
            if not self.passage_encoder.config.pooler_hidden_act:
                self.passage_encoder.config.pooler_hidden_act = "tanh"
            if not self.query_encoder.config.pooler_hidden_act:
                self.query_encoder.config.pooler_hidden_act = "tanh"
            self.passage_pooler = ContextPooler(self.passage_encoder.config)
            self.query_pooler = ContextPooler(self.query_encoder.config)

    def forward(self, batch, type: str = "passage") -> torch.FloatTensor:
        """passage 또는 query를 bert로 encoding합니다."""
        assert type in (
            "passage",
            "query",
        ), "type should be either 'passage' or 'query'"
        output_vectors = []
        if type == "passage":
            last_hidden_states = self.passage_encoder(**batch).last_hidden_state
            if self.passage_pooler:
                output_vector = self.passage_pooler(last_hidden_states)
            else:
                input_mask_expanded = batch["attention_mask"].unsqueeze(-1).expand(last_hidden_states.size()).float()
                sum_embeddings = torch.sum(last_hidden_states * input_mask_expanded, 1)
                sum_mask = input_mask_expanded.sum(1)
                sum_mask = torch.clamp(sum_mask, min=1e-9)
                output_vectors.append(sum_embeddings / sum_mask)
                output_vector = torch.cat(output_vectors, 1)
            return output_vector
        else:
            last_hidden_states = self.query_encoder(**batch).last_hidden_state
            if self.query_pooler:
                output_vector = self.query_pooler(last_hidden_states)
            else:
                input_mask_expanded = batch["attention_mask"].unsqueeze(-1).expand(last_hidden_states.size()).float()
                sum_embeddings = torch.sum(last_hidden_states * input_mask_expanded, 1)
                sum_mask = input_mask_expanded.sum(1)
                sum_mask = torch.clamp(sum_mask, min=1e-9)
                output_vectors.append(sum_embeddings / sum_mask)
                output_vector = torch.cat(output_vectors, 1)
            return output_vector
