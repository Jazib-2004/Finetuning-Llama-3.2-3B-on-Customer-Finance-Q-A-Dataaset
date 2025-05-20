import json
import random
from pathlib import Path
import torch
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training, PeftModel
from huggingface_hub import HfApi, login, create_repo


BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / 'data' / 'processed'
MODEL_NAME = 'meta-llama/Llama-3.2-3B-Instruct'
OUTPUT_DIR = BASE_DIR / 'models' / 'llama-3b-lora-merged'
SEED = 42
BATCH_SIZE = 1
EPOCHS = 3
MAX_LENGTH = 512
HF_REPO_ID = "Syed-Hasan-8503/llama-3b-finetuned-merged"  # Change to your actual repo

def load_faqs(filepath: Path):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def preprocess_examples(examples, tokenizer):
    inputs = []
    for q, a in zip(examples['question'], examples['answer']):
        prompt = f"Question: {q}\nAnswer: {a}"
        inputs.append(prompt)
    model_inputs = tokenizer(inputs, max_length=MAX_LENGTH, truncation=True, padding='max_length')
    model_inputs['labels'] = model_inputs['input_ids'].copy()
    return model_inputs

def main():
    set_seed(SEED)
    faqs = load_faqs(PROCESSED_DIR / 'all_faqs_processed.json')
    random.shuffle(faqs)
    split_idx = int(0.9 * len(faqs))
    train_data, val_data = faqs[:split_idx], faqs[split_idx:]
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
    model = prepare_model_for_kbit_training(model)

    # LoRA config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(model, lora_config)

    train_dataset = train_dataset.map(lambda ex: preprocess_examples(ex, tokenizer), batched=True, remove_columns=['category','question_tokens','answer_tokens'])
    val_dataset = val_dataset.map(lambda ex: preprocess_examples(ex, tokenizer), batched=True, remove_columns=['category','question_tokens','answer_tokens'])

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        save_strategy='epoch',
        logging_steps=1,
        learning_rate=2e-5,
        weight_decay=0.01,
        save_total_limit=2,
        report_to="wandb",  # Change to "wandb" if using it
        fp16=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    trainer.train()

    # Merge LoRA adapters
    model = model.merge_and_unload()
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Upload to Hugging Face Hub
    login(token="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")  # or use `huggingface-cli login` beforehand
    create_repo(HF_REPO_ID, exist_ok=True)
    model.push_to_hub(HF_REPO_ID)
    tokenizer.push_to_hub(HF_REPO_ID)

if __name__ == '__main__':
    main()
