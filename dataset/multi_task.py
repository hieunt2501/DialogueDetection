from ast import literal_eval

import torch

from dataset.utils import tokenize_text


class SequenceLabelingDataset:
    """
    Dataset for text-based dialogue sequence labeling and speaker diarization task
    """
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = literal_eval(self.dataset.iloc[idx]['text'])
        speaker_tag = literal_eval(self.dataset.iloc[idx]['speaker_tag'])
        iob_tag = literal_eval(self.dataset.iloc[idx]['iob_tag'])

        return {
            'text': text,
            'speaker_tag': speaker_tag,
            'iob_tag': iob_tag
        }


class SequenceLabelingCollator(object):
    """
    Dataset collator for text-based dialogue sequence labeling and speaker diarization task
    """
    def __init__(self, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        batch_text = [data["text"] for data in batch]
        speaker_tag = torch.tensor([data["speaker_tag"] for data in batch])
        iob_tag = torch.tensor([data["iob_tag"] for data in batch])

        ids, mask = tokenize_text(self.tokenizer, batch_text, len(batch), self.max_length)

        return {
            "input_ids": ids,
            "attention_mask": mask,
            "speaker_tag": speaker_tag,
            "iob_tag": iob_tag
        }
