from abc import ABC
from typing import Optional, List, cast

import torch
from allennlp.modules.conditional_random_field import ConditionalRandomField, allowed_transitions

from model.base_module import BaseModel, ClassificationHead
from dataset.label_schemes import REVERSE_LABEL


class DialogueDetection(BaseModel, ABC):
    def __init__(self,
                 config,
                 out_dim=512,
                 n_layers=2,
                 dropout=0.1,
                 n_class=4,
                 bidirectional=True,
                 residual=False,
                 crf=False,
                 top_k=1):
        super().__init__(config, out_dim, n_layers, bidirectional)

        # self.window_size = window_size
        self.residual = residual
        self.use_crf = crf
        self.top_k = top_k

        if bidirectional:
            out_dim = 2 * out_dim

        if self.use_crf:
            constraints = allowed_transitions("BIOUL", REVERSE_LABEL)
            self.crf = ConditionalRandomField(num_tags=n_class, constraints=constraints)

        if residual:
            self.transform_linear = torch.nn.Linear(out_dim, self.config.hidden_size)
            self.clf = ClassificationHead(hidden_size=self.config.hidden_size,
                                          n_class=n_class,
                                          dropout=dropout)
        else:
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
                return_dict: Optional[bool] = None,
                tags: Optional[torch.LongTensor] = None):

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
        output = {"logits": iob_logits}

        if self.use_crf:
            mask = torch.ones(outputs.size()[:2], dtype=torch.bool)
            best_paths = self.crf.viterbi_tags(iob_logits, mask, top_k=self.top_k)
            predicted_tags = cast(List[List[int]], [x[0][0] for x in best_paths])
            if tags is not None:
                loss = self.crf(iob_logits, tags, mask)
                output["loss"] = -loss
            output["best_paths"] = best_paths
            output["predicted_tags"] = predicted_tags

        return output
        # return {
            # "logits": iob_logits,
        # }
