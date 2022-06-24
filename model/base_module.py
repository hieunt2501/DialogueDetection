from abc import ABC
from typing import Optional

import torch
from torch import nn
from transformers import RobertaPreTrainedModel, RobertaModel


class BaseModel(RobertaPreTrainedModel, ABC):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self,
                 config,
                 out_dim=512,
                 n_layers=2,
                 bidirectional=True):
        super().__init__(config)

        self.config = config

        self.bert = RobertaModel(config)

        self.lstm = nn.LSTM(input_size=config.hidden_size,
                            hidden_size=out_dim,
                            num_layers=n_layers,
                            bidirectional=bidirectional,
                            batch_first=True)

    def forward(self,
                input_ids: Optional[torch.LongTensor] = None,
                attention_mask: Optional[torch.FloatTensor] = None,
                token_type_ids: Optional[torch.LongTensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                head_mask: Optional[torch.FloatTensor] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None):

        batch_size, _, max_len = input_ids.size()

        outputs = self.bert(
            input_ids.view(-1, max_len),
            attention_mask=attention_mask.view(-1, max_len),
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )[1]  # get pooling output

        outputs = outputs.view(batch_size, -1, outputs.size(-1))
        return outputs


class ClassificationHead(nn.Module):
    def __init__(self, hidden_size, n_class, dropout):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, n_class)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
