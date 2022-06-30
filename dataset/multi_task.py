from ast import literal_eval

import torch
import numpy as np

from dataset.utils import tokenize_text


class SequenceLabelingDataset:
    """
    Dataset for text-based dialogue sequence labeling and speaker diarization task
    """
    def __init__(self, dataset, word_segment=True):
        self.dataset = dataset
        self.word_segment = word_segment

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = literal_eval(self.dataset.iloc[idx]['text'])
        speaker_tag = literal_eval(self.dataset.iloc[idx]['speaker_tag'])
        iob_tag = literal_eval(self.dataset.iloc[idx]['iob_tag'])

        if not self.word_segment:
            text = [" ".join([word.replace("_", " ") for word in sentence.split()]) for sentence in text]

        return {
            'text': text,
            'speaker_tag': speaker_tag,
            'iob_tag': iob_tag
        }


class SequenceLabelingCollator(object):
    """
    Dataset collator for text-based dialogue sequence labeling and speaker diarization task
    """
    def __init__(self, tokenizer=None, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        batch_text = [data["text"] for data in batch]
        speaker_tag = torch.tensor([data["speaker_tag"] for data in batch])
        speaker_tag[speaker_tag == -1] = 2
        iob_tag = torch.tensor([data["iob_tag"] for data in batch])

        if self.tokenizer:
            ids, mask = tokenize_text(self.tokenizer, batch_text, len(batch), self.max_length)

            return {
                "input_ids": ids,
                "attention_mask": mask,
                "speaker_tag": speaker_tag,
                "iob_tag": iob_tag
            }
        else:
            return {
                "text": np.array(batch_text),
                "speaker_tag": speaker_tag,
                "iob_tag": iob_tag
            }
