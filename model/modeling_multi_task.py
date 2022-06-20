from abc import ABC
from typing import Optional

import torch

from model.base_module import BaseModel, ClassificationHead


class BertLSTMSequenceLabeling(BaseModel, ABC):
    def __init__(self,
                 config,
                 out_dim=512,
                 n_layers=2,
                 dropout=0.1,
                 speaker_class=2,
                 iob_class=4,
                 bidirectional=True,
                 window_size=2):
        super().__init__(config, out_dim, n_layers, bidirectional)

        self.window_size = window_size
        if bidirectional:
            out_dim = 2 * out_dim

        self.iob_clf = ClassificationHead(hidden_size=out_dim,
                                          n_class=iob_class,
                                          dropout=dropout)

        self.speaker_clf = ClassificationHead(hidden_size=out_dim,
                                              n_class=speaker_class,
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
        iob_logits = self.iob_clf(lstm_out)

        # speaker diarization
        splits = [outputs[:, i:i + self.window_size, :] for i in range(0, outputs.size(1) - self.window_size + 1)]
        splits = [split.unsqueeze(1) for split in splits]
        splits = torch.cat(splits, dim=1)
        num_samples = splits.size(1)

        splits = splits.view(-1, self.window_size, outputs.size(-1))

        _, (h, _) = self.lstm(splits)
        speaker_logits = torch.cat((h[-1], h[-2]), dim=-1)
        speaker_logits = self.speaker_clf(speaker_logits).view(outputs.size(0), num_samples, -1)

        return {
            "iob_logits": iob_logits,
            "speaker_logits": speaker_logits
        }
