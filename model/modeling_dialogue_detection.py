from abc import ABC
from typing import Optional

import torch

from model.base_module import BaseModel, ClassificationHead


class DialogueDetection(BaseModel, ABC):
    def __init__(self,
                 config,
                 out_dim=512,
                 n_layers=2,
                 dropout=0.1,
                 iob_class=4,
                 bidirectional=True,
                 window_size=2,
                 residual=False):
        super().__init__(config, out_dim, n_layers, bidirectional)

        self.window_size = window_size
        self.residual = residual

        if bidirectional:
            out_dim = 2 * out_dim

        if residual:
            self.transform_linear = torch.nn.Linear(out_dim, self.config.hidden_size)
            self.clf = ClassificationHead(hidden_size=self.config.hidden_size,
                                          n_class=iob_class,
                                          dropout=dropout)
        else:
            self.clf = ClassificationHead(hidden_size=out_dim,
                                          n_class=iob_class,
                                          dropout=dropout)

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

        outputs = super().forward(input_ids,
                                  attention_mask,
                                  token_type_ids,
                                  position_ids,
                                  head_mask,
                                  inputs_embeds,
                                  labels,
                                  output_attentions,
                                  output_hidden_states,
                                  return_dict)

        # iob sequence tagging
        lstm_out, (_, _) = self.lstm(outputs)

        iob_inputs = lstm_out
        if self.residual:
            iob_inputs = self.transform_linear(lstm_out)
            iob_inputs = torch.mean(torch.stack((iob_inputs, outputs)), dim=0)

        iob_logits = self.clf(iob_inputs)

        return {
            "logits": iob_logits,
        }
