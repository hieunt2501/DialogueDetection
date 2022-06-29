import os
import time
import logging
from math import ceil
from collections import defaultdict

import torch
import numpy as np
import sklearn
from torch import nn
from sklearn import metrics
from seqeval.metrics import f1_score, classification_report
from seqeval.scheme import IOBES
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AdamW, get_cosine_with_hard_restarts_schedule_with_warmup

from trainer.amp import MixedPrecisionManager
from utils.tsa_cross_entropy_loss import TsaCrossEntropyLoss
from dataset.label_schemes import REVERSE_LABEL

logger = logging.getLogger(__name__)


class DialogueTrainer:
    def __init__(self, model, args, train_dataset=None, eval_dataset=None, data_collator=None):
        self.args = args
        self.model = model.to(args.use_device)
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator

        if args.resume_from_checkpoint:
            self._load_checkpoint()

        self._init_tensorboard()

    def _save_model(self, optimizer, scheduler, desc=None):
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict()
        }
        output_path = os.path.join(self.args.output_dir, f"model_{desc}.pt")

        torch.save(checkpoint, output_path)

    def _load_checkpoint(self):
        checkpoint = torch.load(self.args.resume_from_checkpoint)
        self.model.load_state_dict(checkpoint['model'])

    def _init_tensorboard(self):
        log_dir = os.path.join(self.args.output_dir, "logs/")
        self.writer = SummaryWriter(log_dir=log_dir)

    @staticmethod
    def _get_loss_fn(num_classes, args=None, total_steps=None, is_eval=False, ignore_index=-100):
        if is_eval or not args.tsa_schedule:
            return nn.CrossEntropyLoss(ignore_index=ignore_index)

        return TsaCrossEntropyLoss(num_steps=total_steps,
                                   num_classes=num_classes,
                                   alpha=args.tsa_alpha,
                                   temperature=args.temperature,
                                   cuda=True if args.use_device == "cuda" else False,
                                   ignore_index=ignore_index)

    @staticmethod
    def _get_optimizer(model, args, total_steps):
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=args.learning_rate,
                          correct_bias=True,
                          eps=args.adam_epsilon,
                          betas=(args.adam_beta1, args.adam_beta2))

        num_warmup_steps = int(total_steps * args.warmup_ratio)
        # scheduler = get_linear_schedule_with_warmup(optimizer,
        #                                             num_warmup_steps=num_warmup_steps,
        #                                             num_training_steps=total_steps)
        scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer,
                                                                       num_warmup_steps=num_warmup_steps,
                                                                       num_training_steps=total_steps,
                                                                       num_cycles=args.num_cycles)
        return optimizer, scheduler

    @staticmethod
    def _get_loader(args, dataset, collator, batch_size):
        loader = DataLoader(dataset,
                            batch_size=batch_size,
                            pin_memory=args.dataloader_pin_memory,
                            num_workers=args.dataloader_num_workers,
                            shuffle=args.shuffle,
                            collate_fn=collator)
        return loader

    @staticmethod
    def _compute_loss(logits, labels, criterion, n_class):
        return criterion(logits.view(-1, n_class), labels.to(logits.device).view(-1))

    def train(self):
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(self.train_dataset)}")
        logger.info(f"  Num Epochs = {self.args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")

        # init data loader
        train_loader = self._get_loader(self.args,
                                        self.train_dataset,
                                        self.data_collator,
                                        self.args.per_device_train_batch_size)
        eval_loader = self._get_loader(self.args,
                                       self.eval_dataset,
                                       self.data_collator,
                                       self.args.per_device_eval_batch_size)

        total_steps = ceil(len(train_loader) * self.args.num_train_epochs / self.args.gradient_accumulation_steps)

        # init optimizer and scheduler
        optimizer, scheduler = self._get_optimizer(self.model, self.args, total_steps)

        # init mixed precision training
        amp = MixedPrecisionManager(self.args.fp16)

        train_loss = 0
        accumulated_steps = 0
        best_eval_loss = float('inf')
        best_eval_score = 0
        start_time = time.time()

        for _ in range(self.args.resume_epoch, int(self.args.num_train_epochs)):
            self.model.train()
            steps_trained_progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
            for idx, batch in steps_trained_progress_bar:
                this_batch_loss = 0
                with amp.context():
                    data = {k: v.to(self.args.use_device) for k, v in batch.items() if "tag" not in k}
                    data["tags"] = batch["iob_tag"]

                    outputs = self.model(**data)
                    loss = outputs["loss"]

                    loss = loss / self.args.temperature / self.args.gradient_accumulation_steps

                    amp.backward(loss)

                    train_loss += loss.item()
                    this_batch_loss += loss.item()
                    accumulated_steps += 1

                if ((idx + 1) % self.args.gradient_accumulation_steps == 0) or (idx + 1) >= len(train_loader):
                    amp.step(self.model, optimizer, scheduler, self.args.max_grad_norm)

                avg_loss = train_loss / accumulated_steps
                elapsed = float(time.time() - start_time)

                self.writer.add_scalar("train/avg_loss", avg_loss, accumulated_steps)
                self.writer.add_scalar("train/batch_loss", this_batch_loss, accumulated_steps)
                self.writer.add_scalar("train/throughput", accumulated_steps / elapsed, accumulated_steps)
                self.writer.add_scalar("train/learning_rate", scheduler.get_last_lr()[0], accumulated_steps)

            eval_loss, eval_score = self.eval(accumulated_steps, eval_loader)

            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                logger.info(f"  Saving best evaluation loss {eval_loss}")
                self._save_model(optimizer, scheduler, desc="best_loss")

            if eval_score > best_eval_score:
                best_eval_score = eval_score
                logger.info(f"  Saving best evaluation f1_score {eval_score}")
                self._save_model(optimizer, scheduler, "best_score")

    def eval(self, accumulated_steps, eval_loader=None):
        logger.info("***** Running evaluation *****")
        logger.info(f"  Num examples = {len(self.eval_dataset)}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_eval_batch_size}")

        if not eval_loader:
            eval_loader = self._get_loader(self.args,
                                           self.eval_dataset,
                                           self.data_collator,
                                           self.args.per_device_eval_batch_size)

        eval_loss = 0
        step_eval_progress_bar = tqdm(enumerate(eval_loader), total=len(eval_loader), leave=False)

        total_raw_pred = []
        total_constraints_pred = []
        total_label = []

        self.model.eval()

        for _, batch in step_eval_progress_bar:
            with torch.no_grad():
                data = {k: v.to(self.args.use_device) for k, v in batch.items() if "tag" not in k}
                data["tags"] = batch["iob_tag"]
                outputs = self.model(**data)

                loss = outputs["loss"]
                logits = outputs["logits"]
                predicted_tags = outputs["predicted_tags"]
                labels = data["tags"]

                raw_pred = logits.argmax(dim=-1).cpu().tolist()
                labels = labels.cpu().tolist()

                raw_pred = [[REVERSE_LABEL[x] for x in p] for p in raw_pred]
                predicted_tags = [[REVERSE_LABEL[x] for x in t] for t in predicted_tags]
                labels = [[REVERSE_LABEL[x] for x in tag] for tag in labels]

                total_raw_pred.extend(raw_pred)
                total_constraints_pred.extend(predicted_tags)
                total_label.extend(labels)

                loss = loss / self.args.temperature / self.args.gradient_accumulation_steps
                eval_loss += loss.item()

        eval_loss /= len(eval_loader)
        self.writer.add_scalar("eval/avg_loss", eval_loss, accumulated_steps)

        label = [x for label in total_label for x in label]

        print("RAW LOGITS PREDICTION")
        raw_f_score = f1_score(total_label, total_raw_pred, average="macro")
        self.writer.add_scalar("eval/raw_f1_score", raw_f_score, accumulated_steps)
        print(classification_report(total_label, total_raw_pred))
        pred = [x for tag in total_raw_pred for x in tag]
        print(sklearn.metrics.classification_report(label, pred))

        print("VITERBI LOGITS PREDICTION")
        viterbi_f_score = f1_score(total_label, total_constraints_pred, average="macro")
        self.writer.add_scalar("eval/viterbi_f1_score", viterbi_f_score, accumulated_steps)
        print(classification_report(total_label, total_constraints_pred))
        pred = [x for tag in total_constraints_pred for x in tag]
        print(sklearn.metrics.classification_report(label, pred))
        # self.writer.add_scalar("eval/f1_score", f_score, accumulated_steps)
        return eval_loss, viterbi_f_score
