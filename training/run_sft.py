#!/usr/bin/env python3
import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

from training.qwen3_no_think import DEFAULT_SYSTEM_PROMPT, tokenize_conversation


def load_json_list(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def preprocess(records: list[dict], tokenizer, max_seq_length: int) -> Dataset:
    rows = []
    for record in records:
        user_text = (record.get("instruction", "") + (record.get("input") or "")).strip()
        encoded = tokenize_conversation(
            tokenizer=tokenizer,
            system=record.get("system") or DEFAULT_SYSTEM_PROMPT,
            user=user_text,
            assistant=record["output"],
            max_seq_length=max_seq_length,
        )
        rows.append(
            {
                "input_ids": encoded.input_ids,
                "attention_mask": encoded.attention_mask,
                "labels": encoded.labels,
                "assistant_mask": encoded.assistant_mask,
            }
        )
    return Dataset.from_list(rows)


@dataclass
class SupervisedDataCollator:
    tokenizer: AutoTokenizer

    def _pad(self, sequences: list[list[int]], pad_value: int) -> torch.Tensor:
        max_len = max(len(seq) for seq in sequences)
        return torch.tensor([seq + [pad_value] * (max_len - len(seq)) for seq in sequences], dtype=torch.long)

    def __call__(self, features):
        input_ids = self._pad([feature["input_ids"] for feature in features], self.tokenizer.pad_token_id)
        attention_mask = self._pad([feature["attention_mask"] for feature in features], 0)
        labels = self._pad([feature["labels"] for feature in features], -100)
        assistant_mask = self._pad([feature["assistant_mask"] for feature in features], 0)
        labels[attention_mask == 0] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "assistant_mask": assistant_mask,
        }


class AssistantOnlyTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        inputs = dict(inputs)
        inputs.pop("assistant_mask", None)
        return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--eval-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-seq-length", type=int, default=1536)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    train_records = load_json_list(Path(args.train_file))
    eval_records = load_json_list(Path(args.eval_file))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[item.strip() for item in args.target_modules.split(",") if item.strip()],
    )
    model = get_peft_model(model, peft_config)

    train_dataset = preprocess(train_records, tokenizer, args.max_seq_length)
    eval_dataset = preprocess(eval_records, tokenizer, args.max_seq_length)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        eval_strategy="steps",
        save_strategy="steps",
        logging_strategy="steps",
        report_to=[],
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
    )

    trainer = AssistantOnlyTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=SupervisedDataCollator(tokenizer),
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    metrics = trainer.evaluate()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output_dir, "eval_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
