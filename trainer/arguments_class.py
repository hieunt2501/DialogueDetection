from typing import Optional
from dataclasses import dataclass, field

from transformers import TrainingArguments


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch"
            )
        }
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_data_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    eval_data_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )


@dataclass
class CustomTrainingArguments(TrainingArguments):
    tsa_schedule: Optional[str] = field(
        default=None, metadata={"help": "Training signal annealing Cross Entropy Loss with schedule (tsa_schedule)."}
    )
    resume_epoch: Optional[int] = field(
        default=0, metadata={"help": "Resume training from epoch (resume_epoch)."}
    )
    shuffle: Optional[bool] = field(
        default=True, metadata={"help": "Shuffle data loader"}
    )
    temperature: Optional[float] = field(
        default=1, metadata={"help": "Softmax temperature"}
    )
    speaker_classes: Optional[int] = field(
        default=2, metadata={"help": "Number of speaker classes"}
    )
    iob_classes: Optional[int] = field(
        default=4, metadata={"help": "Number of iob classes"}
    )
    tsa_alpha: Optional[int] = field(
        default=5, metadata={"help": "Training signal annealing alpha parameter"}
    )
    use_device: Optional[str] = field(
        default="cpu", metadata={"help": "Device for model"}
    )
    num_cycles: Optional[int] = field(
        default=1, metadata={"help": "Number of cycles for hard reset scheduler"}
    )
    task: Optional[str] = field(
        default="multitask", metadata={"help": "Task for training (multitask, dialogue, speaker)"}
    )
    window_size: Optional[int] = field(
        default=2, metadata={"help": "Window size for sliding sentences"}
    )
    lamb_iob: Optional[float] = field(
        default=1, metadata={"help": "Lambda for iob tag loss"}
    )
    lamb_speaker: Optional[float] = field(
        default=1, metadata={"help": "Lambda for speaker tag loss"}
    )
    residual: Optional[bool] = field(
        default=False, metadata={"help": "Residual connection from bert output and lstm output for multi-task"}
    )
    fuse_lstm_information: Optional[bool] = field(
        default=False, metadata={"help": "Fuse information from two lstm output for multi-task"}
    )
    mask_speaker: Optional[bool] = field(
        default=False, metadata={"help": "Mask outside conversation sentences for speaker task"}
    )
    ignore_outside_conversation: Optional[bool] = field(
        default=False, metadata={"help": "Ignore index for outside conversation while computing loss"}
    )
