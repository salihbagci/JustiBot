from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import load_dataset
import torch
import os

# Ortam ayarları
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
torch.cuda.empty_cache()

# Model ve tokenizer
base_model = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.pad_token = tokenizer.eos_token

# Veri setini yükle
dataset = load_dataset("json", data_files="data/train_lawbot.jsonl", split="train")

# Tokenize işlemi
def tokenize(example):
    inputs = tokenizer(example["prompt"], truncation=True, padding="max_length", max_length=512)
    labels = tokenizer(example["response"], truncation=True, padding="max_length", max_length=512)
    inputs["labels"] = labels["input_ids"]
    return inputs

tokenized_dataset = dataset.map(tokenize, remove_columns=["prompt", "response"])

# Modeli yükle ve LoRA için hazırla
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    load_in_4bit=True,
    device_map="auto"
)
model = prepare_model_for_kbit_training(model)

# LoRA ayarları
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)

# Eğitim ayarları
training_args = TrainingArguments(
    output_dir="lawbot-model",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    warmup_steps=10,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    report_to="none"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer
)

# Eğitimi başlat
trainer.train()
model.save_pretrained("lawbot-model")
tokenizer.save_pretrained("lawbot-model")


