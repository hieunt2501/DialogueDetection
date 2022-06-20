from ast import literal_eval

import torch

from dataset.utils import tokenize_text


class SpeakerDiarizationDataset:
    """
    Dataset for text-based speaker diarization task
    """
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sentences = literal_eval(self.dataset.iloc[idx]['sentences'])
        speaker_tag = self.dataset.iloc[idx]['labels']

        return {
            'text': sentences,
            'speaker_tag': speaker_tag
        }


class SpeakerDiarizationCollator(object):
    """
    Dataset collator for text-based speaker diarization task
    """
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        batch_text = [data['text'] for data in batch]
        speaker_tag = torch.tensor([data["speaker_tag"] for data in batch])

        ids, mask = tokenize_text(self.tokenizer, batch_text, len(batch), self.max_length)

        return {
            'input_ids': ids,
            'attention_mask': mask,
            'speaker_tag': speaker_tag
        }
