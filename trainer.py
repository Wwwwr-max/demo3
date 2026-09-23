import contextlib
import math
import os
import random

import numpy as np
import swanlab
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from transformers.optimization import get_scheduler

import dataset
import model_m
from config import load_config
from metrics import NERMetric


class Trainer_t:
    def __init__(self, config=None):
        self.config = config or load_config()
        self.model_config = self.config["model"]
        self.data_config = self.config["data"]
        self.training_config = self.config["training"]

        self.device = self.set_device()
        self.set_seed(self.training_config["seed"])
        self.tokenizer = self.set_tokenizer()

        self.lr_scheduler = None
        self.scaler = None
        self.grad_acc_steps = self.training_config[
            "gradient_accumulation_steps"
        ]
        self.dataloader_num_workers = self.training_config.get(
            "dataloader_num_workers",
            0,
        )
        self.pin_memory = self.training_config.get(
            "pin_memory",
            True,
        )
        self.logging_steps = max(
            1,
            int(
                self.training_config.get(
                    "logging_steps",
                    20,
                )
            ),
        )

        self.use_fp16 = self.training_config["fp16"]
        self.use_bf16 = self.training_config["bf16"]
        self.autocast_dtype = self.get_autocast_dtype()

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def set_device(self):
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def get_autocast_dtype(self):       # 选择要使用的数据类型
        if self.device.type != "cuda":
            return None
        if self.use_bf16:
            return torch.bfloat16
        if self.use_fp16:
            return torch.float16
        return None

    def autocast_context(self):
        if self.autocast_dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(
            device_type=self.device.type,
            dtype=self.autocast_dtype,
        )

    def set_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_config["model_name_or_path"],
            use_fast=False,
            trust_remote_code=self.model_config["trust_remote_code"],
            padding_side="right",       #右侧补padding
        )
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def set_model(self, resume_adapter_path=None):
        model = model_m.Mymodel(
            self.model_config,
            pad_token_id=self.tokenizer.pad_token_id,
            resume_adapter_path=resume_adapter_path,
        )

        if self.model_config["device_map"] is None:
            model.to(self.device)       #移到cuda

        return model

    def set_data(self, split):
        return dataset.Mydataset(
            self.data_config,
            self.tokenizer,
            split=split,
        )

    def set_collate(self):
        return dataset.Mycollator(self.tokenizer)

    def get_gpu_memory_metrics(self):           #记录显存
        if self.device.type != "cuda":
            return {}

        device_index = self.device.index or 0
        return {
            "system/gpu_mem_allocated_gb": (
                torch.cuda.memory_allocated(device_index) / 1024**3
            ),
            "system/gpu_mem_reserved_gb": (
                torch.cuda.memory_reserved(device_index) / 1024**3
            ),
            "system/gpu_mem_max_allocated_gb": (
                torch.cuda.max_memory_allocated(device_index) / 1024**3
            ),
            "system/gpu_mem_total_gb": (
                torch.cuda.get_device_properties(
                    device_index
                ).total_memory / 1024**3
            ),
        }

    def set_lr_scheduler(self, num_steps, optimizer):
        if self.lr_scheduler is None:
            self.lr_scheduler = get_scheduler(
                name = self.training_config["lr_scheduler_type"],
                optimizer=optimizer,
                num_warmup_steps=self.training_config[      #预热步数，从零线性提升上去
                    "num_warmup_steps"
                ],
                num_training_steps=num_steps,       #全部训练步数，用来实现衰减
            )
        return self.lr_scheduler

    def set_train_dataloader(self):
        return DataLoader(
            self.set_data("train"),
            batch_size=self.training_config[
                "per_device_train_batch_size"
            ],
            shuffle=True,
            collate_fn=self.set_collate(),
            num_workers=self.dataloader_num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.dataloader_num_workers > 0,
        )

    def set_dev_dataloader(self):
        return DataLoader(
            self.set_data("eval"),
            batch_size=self.training_config[
                "per_device_eval_batch_size"
            ],
            shuffle=False,
            collate_fn=self.set_collate(),
            num_workers=self.dataloader_num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.dataloader_num_workers > 0,
        )

    def evaluate_loss(self, model, dev_loader):
        was_training = model.training
        model.eval()
        total_eval_loss = 0.0

        with torch.no_grad(), self.autocast_context():
            for batch in dev_loader:
                batch = {
                    key: value.to(self.device)
                    for key, value in batch.items()
                }
                outputs = model(**batch)
                total_eval_loss += outputs.loss.item()

        if was_training:
            model.train()

        return {
            "eval_loss": total_eval_loss / len(dev_loader),
        }

    def evaluate(self, model, dev_loader):
        was_training = model.training
        model.eval()
        total_eval_loss = 0.0
        metric = NERMetric()
        processed_samples = 0
        old_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"        #推理要把padding加在左侧
        progress = tqdm(
            dev_loader,
            desc="Evaluating",
            unit="batch",
            dynamic_ncols=True,
        )
        try:
            with torch.no_grad(), self.autocast_context():
                for batch in progress:
                    batch = {
                        key: value.to(self.device)
                        for key, value in batch.items()
                    }
                    outputs = model(**batch)
                    total_eval_loss += outputs.loss.item()

                    prompt_lengths = []
                    for labels in batch["labels"]:
                        answer_positions = (
                            labels != -100
                        ).nonzero(as_tuple=False)
                        if answer_positions.numel() == 0:
                            prompt_lengths.append(0)
                        else:
                            prompt_lengths.append(
                                int(answer_positions[0].item())
                            )

                    prompt_inputs = [
                        {
                            "input_ids": batch["input_ids"][
                                index,
                                :prompt_lengths[index],
                            ].tolist()
                        }
                        for index in range(batch["input_ids"].shape[0])
                    ]
                    prompt_batch = self.tokenizer.pad(
                        prompt_inputs,
                        padding=True,
                        return_tensors="pt",
                    )
                    prompt_batch = {
                        key: value.to(self.device)
                        for key, value in prompt_batch.items()
                    }

                    generated_ids = model.generate(
                        input_ids=prompt_batch["input_ids"],
                        attention_mask=prompt_batch["attention_mask"],
                        max_new_tokens=self.training_config[        #最大可生成长度
                            "max_new_tokens"
                        ],
                        do_sample=False,        #关闭随机采样
                    )

                    input_length = prompt_batch["input_ids"].shape[1]
                    pred_texts = self.tokenizer.batch_decode(
                        generated_ids[:, input_length:],        #丢掉prompt部分
                        skip_special_tokens=True,
                    )

                    label_mask = batch["labels"] != -100        #布尔型
                    gold_texts = [
                        self.tokenizer.decode(
                            batch["labels"][index][label_mask[index]],      #取出所有非填充
                            skip_special_tokens=True,
                        )
                        for index in range(batch["labels"].shape[0])
                    ]

                    for pred_text, gold_text in zip(
                        pred_texts,
                        gold_texts,
                    ):
                        metric.update(pred_text, gold_text)
                    processed_samples += len(pred_texts)
                    progress.set_postfix(
                        samples=processed_samples,
                    )
        finally:
            progress.close()
            self.tokenizer.padding_side = old_padding_side
            if was_training:
                model.train()

        metrics = metric.compute()
        metrics["eval_loss"] = total_eval_loss / len(dev_loader)
        return metrics

    def save_checkpoint(
        self,
        model,
        optimizer,
        save_path,
        epoch,
        global_step,
    ):
        os.makedirs(save_path, exist_ok=True)
        model.save_pretrained(save_path)

        scaler_state = None
        if self.scaler is not None:
            scaler_state = self.scaler.state_dict()

        checkpoint_state = {
            "epoch": epoch,
            "global_step": global_step,
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": self.lr_scheduler.state_dict(),
            "scaler_state": scaler_state,
        }

        torch.save(
            checkpoint_state,
            os.path.join(save_path, "trainer_state.bin"),
        )

    def load_checkpoint(self, load_path, optimizer):
        state_path = os.path.join(
            load_path,
            "trainer_state.bin",
        )
        state = torch.load(
            state_path,
            map_location=self.device,
        )

        optimizer.load_state_dict(state["optimizer_state"])
        self.lr_scheduler.load_state_dict(state["scheduler_state"])

        if self.scaler is not None and state["scaler_state"] is not None:
            self.scaler.load_state_dict(state["scaler_state"])

        return state["epoch"], state["global_step"]

    def train(self, epochs=None, resume_ckpt=None, output_dir=None):
        epochs = epochs or self.training_config["num_train_epochs"]
        output_dir = (
            output_dir
            or self.training_config.get("output_dir")
            or "./ckpt"
        )
        model = self.set_model(
            resume_adapter_path=resume_ckpt,
        )
        train_loader = self.set_train_dataloader()
        dev_loader = self.set_dev_dataloader()
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),      #只更新需要计算梯度的
            lr=self.training_config["learning_rate"],
            weight_decay=self.training_config["weight_decay"],
        )
        steps_per_epoch = math.ceil(
            len(train_loader) / self.grad_acc_steps
        )
        total_train_steps = steps_per_epoch * epochs
        self.set_lr_scheduler(
            num_steps=total_train_steps,
            optimizer=optimizer,
        )

        if self.use_fp16 and self.device.type == "cuda":
            self.scaler = torch.amp.GradScaler("cuda")

        start_epoch = 0
        global_step = 0
        if resume_ckpt is not None:
            start_epoch, global_step = self.load_checkpoint(
                resume_ckpt,
                optimizer,
            )
            print(
                f"resume epoch={start_epoch}, "
                f"global_step={global_step}"
            )

        grad_clip_norm = self.training_config["max_grad_norm"]

        swanlab.init(
            project=self.training_config["swanlab_project"],
            name=self.training_config["swanlab_exp"],
            config=self.config,
        )

        train_progress = tqdm(
            total=total_train_steps,
            initial=global_step,
            desc="Training",
            unit="step",
            dynamic_ncols=True,
        )

        for epoch in range(start_epoch, epochs):
            train_progress.set_description(
                f"Training epoch {epoch + 1}/{epochs}"
            )
            model.train()
            total_train_loss = torch.zeros((), device=self.device)
            for step_idx, batch in enumerate(train_loader):
                batch = {
                    key: value.to(self.device)
                    for key, value in batch.items()
                }
                with self.autocast_context():
                    outputs = model(**batch)
                    unscaled_loss = outputs.loss
                    loss = unscaled_loss / self.grad_acc_steps
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                total_train_loss += unscaled_loss.detach()
                should_update = (
                    (step_idx + 1) % self.grad_acc_steps == 0
                    or (step_idx + 1) == len(train_loader)
                )

                if not should_update:
                    continue

                if self.scaler is not None:
                    self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    grad_clip_norm,
                )
                if self.scaler is not None:
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    optimizer.step()
                self.lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                train_progress.update(1)
                should_log = (
                    global_step % self.logging_steps == 0
                    or (step_idx + 1) == len(train_loader)
                )
                if should_log:
                    loss_value = float(unscaled_loss.detach())
                    lr_value = self.lr_scheduler.get_last_lr()[0]
                    train_progress.set_postfix(
                        loss=round(loss_value, 4),
                        lr=f"{lr_value:.2e}",
                    )
                    step_log = {
                        "train/loss": loss_value,
                        "train/lr": lr_value,
                        "global_step": global_step,
                    }
                    step_log.update(self.get_gpu_memory_metrics())
                    swanlab.log(step_log)
            avg_train_loss = float(
                total_train_loss / len(train_loader)
            )
            dev_metrics = self.evaluate_loss(model, dev_loader)
            epoch_log = {
                "epoch": epoch + 1,
                "epoch/train_loss": avg_train_loss,
                "epoch/dev_loss": dev_metrics["eval_loss"],
            }
            epoch_log.update(self.get_gpu_memory_metrics())
            swanlab.log(epoch_log)
            print(
                f"epoch={epoch + 1}, "
                f"train_loss={avg_train_loss:.4f}, "
                f"dev_loss={dev_metrics['eval_loss']:.4f}"
            )
            save_path = os.path.join(
                output_dir,
                f"epoch_{epoch + 1}",
            )
            self.save_checkpoint(
                model,
                optimizer,
                save_path,
                epoch + 1,
                global_step,
            )
            print(f"saved checkpoint to {save_path}")

        train_progress.close()
        final_dev_metrics = self.evaluate(model, dev_loader)
        print(
            f"final dev: "
            f"loss={final_dev_metrics['eval_loss']:.4f}, "
            f"P={final_dev_metrics['precision']:.4f}, "
            f"R={final_dev_metrics['recall']:.4f}, "
            f"F1={final_dev_metrics['f1']:.4f}"
        )

        swanlab.finish()
