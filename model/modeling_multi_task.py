from abc import ABC
from typing import Optional

import torch

from model.base_module import BaseModel, ClassificationHead


def splits_into_pairs(x: torch.Tensor, window_size=2):
    splits = [x[:, i:i + window_size, :] for i in range(0, x.size(1) - window_size + 1)]
    splits = [split.unsqueeze(1) for split in splits]
    splits = torch.cat(splits, dim=1)
    return splits


class BertLSTMSequenceLabeling(BaseModel, ABC):
    def __init__(self,
                 config,
                 out_dim=512,
                 n_layers=2,
                 dropout=0.1,
                 speaker_class=2,
                 iob_class=4,
                 bidirectional=True,
                 window_size=2,
                 fuse_lstm_information=False,
                 residual=False):
        super().__init__(config, out_dim, n_layers, bidirectional)

        self.window_size = window_size
        self.fuse_lstm_information = fuse_lstm_information
        self.residual = residual

        if bidirectional:
            out_dim = 2 * out_dim

        if residual:
            self.transform_linear = torch.nn.Linear(out_dim, self.config.hidden_size)

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
                return_dict: Optional[bool] = None,
                speaker_tag: Optional[torch.LongTensor] = None):

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
        speaker_inputs = outputs
        if self.residual:
            speaker_inputs = self.transform_linear(lstm_out)
            speaker_inputs = torch.mean(torch.stack((speaker_inputs, outputs)), dim=0)

        speaker_inputs = splits_into_pairs(speaker_inputs, self.window_size)

        if speaker_tag is not None:
            mask = speaker_tag != -1
            mask = mask.unsqueeze(-1).unsqueeze(-1).expand(speaker_inputs.size())
            speaker_inputs *= mask

        speaker_inputs = speaker_inputs.view(-1, self.window_size, outputs.size(-1))

        _, (h, _) = self.lstm(speaker_inputs)
        speaker_logits = torch.cat((h[-1], h[-2]), dim=-1).view(outputs.size(0),
                                                                outputs.size(1) - self.window_size + 1,
                                                                -1)

        if self.fuse_lstm_information:
            # batch_size x num_pairs x window_size x lstm output dim
            iob_lstm_splits = splits_into_pairs(lstm_out, self.window_size)
            iob_lstm_splits = torch.mean(iob_lstm_splits, dim=2)
            speaker_logits = torch.mean(torch.stack((iob_lstm_splits, speaker_logits)), dim=0)

        speaker_logits = self.speaker_clf(speaker_logits)

        return {
            "iob_logits": iob_logits,
            "speaker_logits": speaker_logits
        }
