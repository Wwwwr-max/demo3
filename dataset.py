import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from config import load_config


class Mydataset(Dataset):
    def __init__(self, data_config, tokenizer, split="train", max_length=None):
        data_config = data_config or load_config()["data"]
        self.data_path = (
            Path(data_config["root_dir"])
            / data_config[f"{split}_file"]
        )
        self.tokenizer = tokenizer
        self.max_length = (
            data_config["max_seq_length"]
            if max_length is None
            else max_length
        )
        self.system_prompt = (
            "You are a biomedical NER system. "
            "Extract all GENE entities. "
            "Output one entity per line as name:GENE."
            "If there are no entities, output 无实体."
        )
        with open(self.data_path,"r", encoding="utf-8") as f:
            self.items = json.load(f)
        self.samples = [
            self.encode_item(item)
            for item in self.items
        ]
    def encode_item(self, item):
        messages = [
            {
                "role": "system",
                "content": self.system_prompt,
            },
            {
                "role": "user",
                "content": item["input"],
            },
        ]
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        full_messages = messages + [
            {
                "role": "assistant",
                "content": item["output"],
            },
        ]

        full_text = self.tokenizer.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        prompt_ids = self.tokenizer(
            prompt_text,
            add_special_tokens=False,
        )["input_ids"]

        full_ids = self.tokenizer(
            full_text,
            add_special_tokens=False,
        )["input_ids"]

        target_ids = full_ids[len(prompt_ids):]

        if len(target_ids) >= self.max_length:
            target_ids = target_ids[:self.max_length]
            prompt_ids = []
        elif len(prompt_ids) + len(target_ids) > self.max_length:
            prompt_budget = self.max_length - len(target_ids)
            prompt_ids = (
                prompt_ids[-prompt_budget:]
                if prompt_budget > 0
                else []
            )

        input_ids = prompt_ids + target_ids
        attention_mask = [1] * len(input_ids)
        labels = [-100] * len(prompt_ids) + target_ids

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(
                attention_mask,
                dtype=torch.long,
            ),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]

class Mycollator():
    def __init__(self,tokenizer):
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, features):
        max_length = 0
        for feature in features:
            length = len(feature["input_ids"])
            if length>=max_length:
                max_length = length

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        for feature in features:
            input_ids = feature["input_ids"].tolist()
            attention_mask = feature["attention_mask"].tolist()
            labels = feature["labels"].tolist()
            pad_len = max_length - len(input_ids)

            input_ids = input_ids+[self.pad_token_id]*pad_len
            attention_mask = attention_mask+[0]*pad_len
            labels = labels+[-100]*pad_len
            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels)
        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }
