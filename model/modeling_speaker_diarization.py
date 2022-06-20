from abc import ABC
from typing import Optional

import torch

from model.base_module import BaseModel, ClassificationHead


class SpeakerDiarization(BaseModel, ABC):
    def __init__(self,
                 config,
                 out_dim=512,
                 n_class=2,
                 n_layers=2,
                 bidirectional=True,
                 dropout=0.1,
                 window_size=2):
        super().__init__(config, out_dim, n_layers, bidirectional)

        if bidirectional:
            out_dim *= 2
        self.window_size = window_size
        self.clf = ClassificationHead(hidden_size=out_dim,
                                      n_class=n_class,
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

        # speaker diarization
        _, (h, _) = self.lstm(outputs)
        speaker_logits = torch.cat((h[-1], h[-2]), dim=-1)
        speaker_logits = self.clf(speaker_logits)

        return {
            "logits": speaker_logits
        }
