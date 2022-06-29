from abc import ABC
from typing import Optional, List, Tuple, Dict, cast

import torch
from allennlp.modules.conditional_random_field import ConditionalRandomField
from allennlp.common.checks import ConfigurationError

from model.base_module import BaseModel, ClassificationHead
from dataset.label_schemes import get_reverse_label


def allowed_transitions(constraint_type: str, labels: Dict[int, str]) -> List[Tuple[int, int]]:
    """
    Given labels and a constraint type, returns the allowed transitions. It will
    additionally include transitions for the start and end states, which are used
    by the conditional random field.

    # Parameters

    constraint_type : `str`, required
        Indicates which constraint to apply. Current choices are
        "BIO", "IOB1", "BIOUL", and "BMES".
    labels : `Dict[int, str]`, required
        A mapping {label_id -> label}. Most commonly this would be the value from
        Vocabulary.get_index_to_token_vocabulary()

    # Returns

    `List[Tuple[int, int]]`
        The allowed transitions (from_label_id, to_label_id).
    """
    num_labels = len(labels)
    start_tag = num_labels
    end_tag = num_labels + 1
    labels_with_boundaries = list(labels.items()) + [(start_tag, "START"), (end_tag, "END")]

    allowed = []
    for from_label_index, from_label in labels_with_boundaries:
        if from_label in ("START", "END"):
            from_tag = from_label
            from_entity = ""
        else:
            from_tag = from_label[0]
            from_entity = from_label[1:]
        for to_label_index, to_label in labels_with_boundaries:
            if to_label in ("START", "END"):
                to_tag = to_label
                to_entity = ""
            else:
                to_tag = to_label[0]
                to_entity = to_label[1:]
            if is_transition_allowed(constraint_type, from_tag, from_entity, to_tag, to_entity):
                allowed.append((from_label_index, to_label_index))
    return allowed


def is_transition_allowed(
    constraint_type: str, from_tag: str, from_entity: str, to_tag: str, to_entity: str
):
    """
    Given a constraint type and strings `from_tag` and `to_tag` that
    represent the origin and destination of the transition, return whether
    the transition is allowed under the given constraint type.

    # Parameters

    constraint_type : `str`, required
        Indicates which constraint to apply. Current choices are
        "BIO", "IOB1", "BIOUL", and "BMES".
    from_tag : `str`, required
        The tag that the transition originates from. For example, if the
        label is `I-PER`, the `from_tag` is `I`.
    from_entity : `str`, required
        The entity corresponding to the `from_tag`. For example, if the
        label is `I-PER`, the `from_entity` is `PER`.
    to_tag : `str`, required
        The tag that the transition leads to. For example, if the
        label is `I-PER`, the `to_tag` is `I`.
    to_entity : `str`, required
        The entity corresponding to the `to_tag`. For example, if the
        label is `I-PER`, the `to_entity` is `PER`.

    # Returns

    `bool`
        Whether the transition is allowed under the given `constraint_type`.
    """

    if to_tag == "START" or from_tag == "END":
        # Cannot transition into START or from END
        return False

    if constraint_type == "BIOUL":
        if from_tag == "START":
            return to_tag in ("O", "B", "U")
        if to_tag == "END":
            return from_tag in ("O", "L", "U")
        return any(
            [
                # O can transition to O, B-* or U-*
                # L-x can transition to O, B-*, or U-*
                # U-x can transition to O, B-*, or U-*
                from_tag in ("O", "L", "U") and to_tag in ("O", "B", "U"),
                # B-x can only transition to I-x or L-x
                # I-x can only transition to I-x or L-x
                from_tag in ("B", "I") and to_tag in ("I", "L") and from_entity == to_entity,
            ]
        )
    elif constraint_type == "BIOL":
        if from_tag == "START":
            return to_tag in ("O", "B")
        if to_tag == "END":
            return from_tag in ("O", "L")
        return any(
            [
                # O can transition to O, B-*
                # L-x can transition to O, B-*,
                from_tag in ("O", "L") and to_tag in ("O", "B"),
                # B-x can only transition to I-x or L-x
                # I-x can only transition to I-x or L-x
                from_tag in ("B", "I") and to_tag in ("I", "L") and from_entity == to_entity,
            ]
        )
    elif constraint_type == "BIO":
        if from_tag == "START":
            return to_tag in ("O", "B")
        if to_tag == "END":
            return from_tag in ("O", "B", "I")
        return any(
            [
                # Can always transition to O or B-x
                to_tag in ("O", "B"),
                # Can only transition to I-x from B-x or I-x
                to_tag == "I" and from_tag in ("B", "I") and from_entity == to_entity,
            ]
        )
    elif constraint_type == "IOB1":
        if from_tag == "START":
            return to_tag in ("O", "I")
        if to_tag == "END":
            return from_tag in ("O", "B", "I")
        return any(
            [
                # Can always transition to O or I-x
                to_tag in ("O", "I"),
                # Can only transition to B-x from B-x or I-x, where
                # x is the same tag.
                to_tag == "B" and from_tag in ("B", "I") and from_entity == to_entity,
            ]
        )
    elif constraint_type == "BMES":
        if from_tag == "START":
            return to_tag in ("B", "S")
        if to_tag == "END":
            return from_tag in ("E", "S")
        return any(
            [
                # Can only transition to B or S from E or S.
                to_tag in ("B", "S") and from_tag in ("E", "S"),
                # Can only transition to M-x from B-x, where
                # x is the same tag.
                to_tag == "M" and from_tag in ("B", "M") and from_entity == to_entity,
                # Can only transition to E-x from B-x or M-x, where
                # x is the same tag.
                to_tag == "E" and from_tag in ("B", "M") and from_entity == to_entity,
            ]
        )
    else:
        raise ConfigurationError(f"Unknown constraint type: {constraint_type}")


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
            reverse_label = get_reverse_label(n_class)
            if n_class == 4:
                constraints = allowed_transitions("BIOL", reverse_label)
            else:
                constraints = allowed_transitions("BIOUL", reverse_label)
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
            mask = torch.ones(outputs.size()[:2], dtype=torch.bool).to(outputs.device)
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
