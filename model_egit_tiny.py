from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import load_dataset
import torch
import os

# Ortam ayarları
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
torch.cuda.empty_cache()

# TinyLlama modeli
base_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.pad_token = tokenizer.eos_token

# Veri kümesini yükle (temizlenmiş, chat formatında)
dataset = load_dataset("json", data_files="data/cleaned_chat_format.jsonl", split="train")

# Tokenize fonksiyonu – prompt + response birleştirme
def tokenize(example):
    full_text = example["prompt"] + example["response"] + "</s>"
    tokenized = tokenizer(
        full_text,
        truncation=True,
        padding="max_length",
        max_length=512
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_dataset = dataset.map(tokenize, remove_columns=["prompt", "response"])

# Modeli yükle ve LoRA ile hazırla
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    load_in_4bit=True,
    device_map="auto"
)
model = prepare_model_for_kbit_training(model)

# LoRA ayarları
lora_config = LoraConfig(
    r=4,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)

# Eğitim ayarları
training_args = TrainingArguments(
    output_dir="tiny-lawbot-model",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=20,
    save_steps=500,
    save_total_limit=1,
    report_to="none"
)

# Trainer başlat
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer
)

# Eğitimi başlat
trainer.train()

# Model ve tokenizer'ı kaydet
model.save_pretrained("tiny-lawbot-model")
tokenizer.save_pretrained("tiny-lawbot-model")

