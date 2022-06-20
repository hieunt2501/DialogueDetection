speaker_label_dict = {
    ('O', 'O'): 0,
    ('S1', 'O'): 0,
    ('S2', 'O'): 0,
    ('O', 'S1'): 0,
    ('O', 'S2'): 0,
    ('S1', 'S1'): 0,
    ('S2', 'S2'): 0,
    ('S1', 'S2'): 1,
    ('S2', 'S1'): 1
}


iob_tags = {
    "O": 0,
    "B-D": 1,
    "I-D": 2,
    "E-D": 3
}


def process_speaker_tags(tags):
    """
    Process speaker change tag for each sentence pairs
    :param tags: Speaker tags
    :return: Process speaker tags
    """
    final_tags = [speaker_label_dict[tags[idx], tags[idx + 1]] for idx in range(len(tags) - 1)]
    return final_tags


def process_iob_tags(tags):
    """
    Process IOB tags for each sample
    :param tags: IOB tags
    :return: Processed IOB tags
    """
    if tags[0] == "I-D":
        tags[0] = "B-D"

    if tags[-1] == "I-D":
        tags[-1] = "E-D"

    return [iob_tags[tag] for tag in tags]


def tokenize_text(tokenizer, batch_text, batch_size, max_length):
    """
    Tokenize tex for data loader
    :param tokenizer: pretrained tokenizer
    :param batch_text: text of batch
    :param batch_size: batch size
    :param max_length: maximum length for padding and truncate
    :return: input ids and attention mask
    """
    num_sentences = len(batch_text[0])

    sentences = tokenizer([sentence for sentences in batch_text for sentence in sentences],
                          padding='longest', truncation='longest_first',
                          return_tensors='pt', max_length=max_length)
    ids, mask = sentences['input_ids'], sentences['attention_mask']

    ids = ids.view(batch_size, num_sentences, ids.size(-1))
    mask = mask.view(batch_size, num_sentences, mask.size(-1))

    return ids, mask
