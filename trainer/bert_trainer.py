import os
import time
import logging
from math import ceil
from collections import defaultdict

import torch
import numpy as np
from torch import nn
from sklearn import metrics
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AdamW, get_cosine_with_hard_restarts_schedule_with_warmup

from trainer.amp import MixedPrecisionManager
from utils.tsa_cross_entropy_loss import TsaCrossEntropyLoss

logger = logging.getLogger(__name__)


class Trainer:
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

        # init loss function
        loss_fn = dict()
        if self.args.task == "multitask":
            iob_loss_fn = self._get_loss_fn(self.args.iob_classes, self.args, total_steps)
            speaker_loss_fn = self._get_loss_fn(self.args.speaker_classes,
                                                self.args,
                                                total_steps,
                                                ignore_index=-1 if self.args.ignore_outside_conversation else -100)

            loss_fn["iob_loss_fn"] = iob_loss_fn
            loss_fn["speaker_loss_fn"] = speaker_loss_fn
        elif self.args.task == "speaker":
            loss_fn["loss_fn"] = self._get_loss_fn(self.args.speaker_classes, self.args, total_steps)
        else:
            loss_fn["loss_fn"] = self._get_loss_fn(self.args.iob_classes, self.args, total_steps)

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
                    if self.args.task == "multitask" and self.args.mask_speaker:
                        data["speaker_tag"] = batch["speaker_tag"].to(self.args.use_device)
                    outputs = self.model(**data)

                    if self.args.task == "multitask":
                        loss, _, _ = self._train_step_multitask(batch,
                                                                outputs,
                                                                loss_fn["iob_loss_fn"],
                                                                loss_fn["speaker_loss_fn"])
                    else:
                        loss, _, _ = self._train_step(batch, outputs, loss_fn["loss_fn"])

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

            if self.args.task == "multitask":
                f1_iob, f1_speaker = eval_score
                eval_score = np.mean((f1_iob, f1_speaker))

            if eval_score > best_eval_score:
                best_eval_score = eval_score
                logger.info(f"  Saving best evaluation f1_score {eval_score}")
                self._save_model(optimizer, scheduler, "best_score")

    def _train_step(self, batch, outputs, loss_fn):
        logits = outputs["logits"]
        if self.args.task == "speaker":
            labels = batch["speaker_tag"]
            loss = self._compute_loss(logits=logits,
                                      labels=labels,
                                      criterion=loss_fn,
                                      n_class=self.args.speaker_classes)
        else:
            labels = batch["iob_tag"]
            loss = self._compute_loss(logits=logits,
                                      labels=labels,
                                      criterion=loss_fn,
                                      n_class=self.args.iob_classes)

        return loss, logits, labels

    def _train_step_multitask(self, batch, outputs, iob_loss_fn, speaker_loss_fn):
        iob_logits = outputs["iob_logits"]
        speaker_logits = outputs["speaker_logits"]

        iob_tag = batch["iob_tag"]
        speaker_tag = batch["speaker_tag"]

        iob_loss = self._compute_loss(logits=iob_logits,
                                      labels=iob_tag,
                                      criterion=iob_loss_fn,
                                      n_class=self.args.iob_classes)
        speaker_loss = self._compute_loss(logits=speaker_logits,
                                          labels=speaker_tag,
                                          criterion=speaker_loss_fn,
                                          n_class=self.args.speaker_classes)

        loss = self.args.lamb_iob * iob_loss + self.args.lamb_speaker * speaker_loss

        return loss, (iob_logits, speaker_logits), (iob_tag, speaker_tag)

    def eval(self, accumulated_steps, eval_loader=None):
        logger.info("***** Running evaluation *****")
        logger.info(f"  Num examples = {len(self.eval_dataset)}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_eval_batch_size}")

        if not eval_loader:
            eval_loader = self._get_loader(self.args, self.eval_dataset, self.data_collator)

        eval_loss = 0
        eval_loss_fn = dict()
        if self.args.task == "multitask":
            eval_iob_loss_fn = self._get_loss_fn(self.args.iob_classes, is_eval=True)
            eval_speaker_loss_fn = self._get_loss_fn(self.args.speaker_classes,
                                                     is_eval=True,
                                                     ignore_index=-1 if self.args.ignore_outside_conversation else -100)

            eval_loss_fn["eval_iob_loss_fn"] = eval_iob_loss_fn
            eval_loss_fn["eval_speaker_loss_fn"] = eval_speaker_loss_fn
        elif self.args.task == "speaker":
            eval_loss_fn["loss_fn"] = self._get_loss_fn(self.args.speaker_classes, is_eval=True)
        else:
            eval_loss_fn["loss_fn"] = self._get_loss_fn(self.args.iob_classes, is_eval=True)

        step_eval_progress_bar = tqdm(enumerate(eval_loader), total=len(eval_loader), leave=False)

        total_pred = defaultdict(list)
        total_label = defaultdict(list)
        self.model.eval()

        for _, batch in step_eval_progress_bar:
            with torch.no_grad():
                data = {k: v.to(self.args.use_device) for k, v in batch.items() if "tag" not in k}
                outputs = self.model(**data)

                if self.args.task == "multitask":
                    loss, logits, labels = self._train_step_multitask(batch,
                                                                      outputs,
                                                                      eval_loss_fn["eval_iob_loss_fn"],
                                                                      eval_loss_fn["eval_speaker_loss_fn"])

                    iob_pred = logits[0].argmax(dim=-1).cpu().view(-1).tolist()
                    speaker_pred = logits[1].argmax(dim=-1).cpu().view(-1).tolist()

                    iob_tag = labels[0].cpu().view(-1).tolist()
                    speaker_tag = labels[1].cpu().view(-1).tolist()

                    total_pred["total_iob_pred"].extend(iob_pred)
                    total_pred["total_speaker_pred"].extend(speaker_pred)

                    total_label["total_iob_tag"].extend(iob_tag)
                    total_label["total_speaker_tag"].extend(speaker_tag)
                else:
                    loss, logits, labels = self._train_step(batch, outputs, eval_loss_fn["loss_fn"])

                    pred = logits.argmax(dim=-1).cpu().view(-1).tolist()
                    labels = labels.cpu().view(-1).tolist()

                    total_pred["logits"].extend(pred)
                    total_label["tag"].extend(labels)

                loss = loss / self.args.temperature / self.args.gradient_accumulation_steps
                eval_loss += loss.item()

        eval_loss /= len(eval_loader)
        self.writer.add_scalar("eval/avg_loss", eval_loss, accumulated_steps)

        # if self.args.multitask:
        if self.args.task == "multitask":
            iob_f_score = metrics.f1_score(total_label["total_iob_tag"],
                                           total_pred["total_iob_pred"],
                                           average="macro")
            speaker_f_score = metrics.f1_score(total_label["total_speaker_tag"],
                                               total_pred["total_speaker_pred"],
                                               average="macro")
            self.writer.add_scalar("eval/iob_f1_score", iob_f_score, accumulated_steps)
            self.writer.add_scalar("eval/speaker_f1_score", speaker_f_score, accumulated_steps)

            logger.info(f"  Classification report: IOB Tag")
            print(metrics.classification_report(total_label["total_iob_tag"], total_pred["total_iob_pred"]))
            logger.info(f"  Classification report: Speaker Tag")
            print(metrics.classification_report(total_label["total_speaker_tag"], total_pred["total_speaker_pred"]))

            return eval_loss, (iob_f_score, speaker_f_score)
        else:
            f_score = metrics.f1_score(total_label["tag"], total_pred["logits"], average="macro")
            self.writer.add_scalar("eval/f1_score", f_score, accumulated_steps)
            print(metrics.classification_report(total_label["tag"], total_pred["logits"]))
            return eval_loss, f_score
