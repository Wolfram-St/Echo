import os
import json
from dataclasses import dataclass
from typing import List, Dict

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Minimal QLoRA style fine-tune for small JSONL instruction dataset

MODEL_NAME = os.environ.get("BASE_MODEL", "google/gemma-2b-it")
DATA_FILE = os.environ.get("DATA_FILE", "data/wellness_dataset_sample.jsonl")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./lora-out")

@dataclass
class Record:
    instruction: str
    output: str


def load_local_jsonl(path: str) -> List[Record]:
    rows: List[Record] = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if 'instruction' in obj and 'output' in obj:
                rows.append(Record(obj['instruction'], obj['output']))
    return rows


def build_prompt(example: Record) -> str:
    # Simple chat style prompt; gemma uses <start_of_turn> etc in official format, but we'll keep minimal
    return f"User: {example.instruction}\nAssistant: {example.output}"  # could extend with system preamble


def main():
    device_map = 'auto'

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        load_in_4bit=True,
        device_map=device_map,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)

    records = load_local_jsonl(DATA_FILE)
    print(f"Loaded {len(records)} records")

    texts = [build_prompt(r) for r in records]

    ds = {"text": texts}
    # Use Dataset.from_dict to avoid external requirements
    from datasets import Dataset
    dataset = Dataset.from_dict(ds)

    def tokenize(examples: Dict[str, List[str]]):
        return tokenizer(examples["text"], truncation=True, max_length=512)

    tokenized = dataset.map(tokenize, batched=True)

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    args = TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=torch.cuda.is_available(),
        logging_steps=5,
        output_dir=OUTPUT_DIR,
        save_strategy="no",
        report_to=[]
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )

    print("Starting training...")
    trainer.train()

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Saved LoRA adapter to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
