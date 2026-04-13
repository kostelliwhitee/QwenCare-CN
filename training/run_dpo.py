#!/usr/bin/env python3
import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import Dataset
from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

from training.qwen3_no_think import DEFAULT_SYSTEM_PROMPT, build_chat_messages, render_chat


def load_json_list(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_chat_prompt(tokenizer, prompt: str) -> str:
    return render_chat(
        build_chat_messages(system=DEFAULT_SYSTEM_PROMPT, user=prompt),
        add_generation_prompt=True,
        no_think=True,
    )


def tokenize_pair(tokenizer, prompt: str, response: str, max_length: int) -> dict:
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(response, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
    input_ids = (prompt_ids + response_ids)[:max_length]
    attention_mask = [1] * len(input_ids)
    response_mask = ([0] * min(len(prompt_ids), len(input_ids)) + [1] * max(len(input_ids) - len(prompt_ids), 0))[:max_length]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "response_mask": response_mask,
    }


def preprocess(records: list[dict], tokenizer, max_length: int) -> Dataset:
    rows = []
    for item in records:
        prompt = build_chat_prompt(tokenizer, item["prompt"])
        chosen = tokenize_pair(tokenizer, prompt, item["chosen"], max_length)
        rejected = tokenize_pair(tokenizer, prompt, item["rejected"], max_length)
        rows.append(
            {
                "chosen_input_ids": chosen["input_ids"],
                "chosen_attention_mask": chosen["attention_mask"],
                "chosen_response_mask": chosen["response_mask"],
                "rejected_input_ids": rejected["input_ids"],
                "rejected_attention_mask": rejected["attention_mask"],
                "rejected_response_mask": rejected["response_mask"],
            }
        )
    return Dataset.from_list(rows)


@dataclass
class PreferenceCollator:
    tokenizer: AutoTokenizer

    def _pad_side(self, seqs, pad_value):
        max_len = max(len(seq) for seq in seqs)
        return torch.tensor([seq + [pad_value] * (max_len - len(seq)) for seq in seqs], dtype=torch.long)

    def __call__(self, features):
        return {
            "chosen_input_ids": self._pad_side([f["chosen_input_ids"] for f in features], self.tokenizer.pad_token_id),
            "chosen_attention_mask": self._pad_side([f["chosen_attention_mask"] for f in features], 0),
            "chosen_response_mask": self._pad_side([f["chosen_response_mask"] for f in features], 0),
            "rejected_input_ids": self._pad_side([f["rejected_input_ids"] for f in features], self.tokenizer.pad_token_id),
            "rejected_attention_mask": self._pad_side([f["rejected_attention_mask"] for f in features], 0),
            "rejected_response_mask": self._pad_side([f["rejected_response_mask"] for f in features], 0),
        }


def sequence_logprob(model, input_ids, attention_mask, response_mask):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]
    labels = input_ids[:, 1:]
    token_mask = response_mask[:, 1:].float()
    log_probs = F.log_softmax(logits, dim=-1)
    gathered = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return (gathered * token_mask).sum(dim=-1)


class DPOTrainerLite(Trainer):
    def __init__(self, ref_model, beta: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ref_model = ref_model
        self.beta = beta

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        chosen_input_ids = inputs["chosen_input_ids"]
        chosen_attention_mask = inputs["chosen_attention_mask"]
        chosen_response_mask = inputs["chosen_response_mask"]
        rejected_input_ids = inputs["rejected_input_ids"]
        rejected_attention_mask = inputs["rejected_attention_mask"]
        rejected_response_mask = inputs["rejected_response_mask"]

        policy_chosen = sequence_logprob(model, chosen_input_ids, chosen_attention_mask, chosen_response_mask)
        policy_rejected = sequence_logprob(model, rejected_input_ids, rejected_attention_mask, rejected_response_mask)

        with torch.no_grad():
            ref_chosen = sequence_logprob(self.ref_model, chosen_input_ids, chosen_attention_mask, chosen_response_mask)
            ref_rejected = sequence_logprob(self.ref_model, rejected_input_ids, rejected_attention_mask, rejected_response_mask)

        logits = (policy_chosen - policy_rejected) - (ref_chosen - ref_rejected)
        loss = -F.logsigmoid(self.beta * logits).mean()
        return (loss, {"logits": logits}) if return_outputs else loss


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-name-or-path", required=True)
    parser.add_argument("--sft-adapter-path", required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--beta", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    policy_model = AutoPeftModelForCausalLM.from_pretrained(
        args.sft_adapter_path,
        is_trainable=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    ).to(device)
    ref_model = AutoPeftModelForCausalLM.from_pretrained(
        args.sft_adapter_path,
        is_trainable=False,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    ).to(device)
    policy_model.config.use_cache = False
    ref_model.config.use_cache = False
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    data = load_json_list(Path(args.train_file))
    dataset = preprocess(data, tokenizer, args.max_length)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        logging_strategy="steps",
        save_strategy="steps",
        report_to=[],
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
    )

    trainer = DPOTrainerLite(
        model=policy_model,
        ref_model=ref_model,
        beta=args.beta,
        args=training_args,
        train_dataset=dataset,
        data_collator=PreferenceCollator(tokenizer),
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if trainer.state.log_history:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        Path(args.output_dir, "trainer_state.json").write_text(json.dumps(trainer.state.log_history, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"output_dir": args.output_dir, "rows": len(dataset)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
