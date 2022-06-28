import os
import logging

import transformers
import pandas as pd
from sklearn.model_selection import train_test_split

from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    set_seed
)

from trainer.arguments_class import DataTrainingArguments, ModelArguments, CustomTrainingArguments
from model.modeling_multi_task import BertLSTMSequenceLabeling
from model.modeling_speaker_diarization import SpeakerDiarization
from model.modeling_dialogue_detection import DialogueDetection
from dataset.multi_task import SequenceLabelingDataset, SequenceLabelingCollator
from dataset.speaker_diarization import SpeakerDiarizationDataset, SpeakerDiarizationCollator
from trainer.bert_trainer import Trainer

logger = logging.getLogger(__name__)


def get_dataset(data_args, dataset):
    train_dataframe = pd.read_csv(data_args.train_data_file, encoding='utf8')

    if data_args.eval_data_file:
        eval_dataframe = pd.read_csv(data_args.eval_data_file, encoding='utf8')
    else:
        train_dataframe, eval_dataframe = train_test_split(train_dataframe, test_size=0.2, random_state=42)

    train_dataset = dataset(train_dataframe)
    eval_dataset = dataset(eval_dataframe)

    return train_dataset, eval_dataset


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, CustomTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exist and is not empty. Use"
            " --overwrite_output_dir to overcome"
        )

    # Set seed
    set_seed(training_args.seed)

    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, cache_dir=model_args.cache_dir)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    else:
        raise ValueError(
            "Please provide a pretrained tokenizer name"
        )

    task = training_args.task

    if model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    else:
        config = AutoConfig.from_pretrained('vinai/phobert-base')

    if task == "multitask":
        model = BertLSTMSequenceLabeling(config,
                                         fuse_lstm_information=training_args.fuse_lstm_information,
                                         residual=training_args.residual,
                                         speaker_class=training_args.speaker_classes,
                                         iob_class=training_args.iob_classes)
        collator = SequenceLabelingCollator(tokenizer)
        dataset = SequenceLabelingDataset
    elif task == "speaker":
        model = SpeakerDiarization(config,
                                   speaker_class=training_args.speaker_classes)
        collator = SpeakerDiarizationCollator(tokenizer)
        dataset = SpeakerDiarizationDataset
    else:
        model = DialogueDetection(config,
                                  iob_class=training_args.iob_classes,
                                  residual=training_args.residual)
        collator = SequenceLabelingCollator(tokenizer)
        dataset = SequenceLabelingDataset

    if not data_args.eval_data_file and not data_args.train_data_file:
        raise ValueError("Please provide train and eval data file")
    else:
        train_dataset, eval_dataset = get_dataset(data_args, dataset)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator
    )
    trainer.train()


if __name__ == "__main__":
    main()
