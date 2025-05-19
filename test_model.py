from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Modeli ve tokenizer'ı yükle
model_path = "tiny-lawbot-model"
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Test prompt
prompt = "<|user|>\nAnayasa madde 2'ye göre Türkiye Cumhuriyeti'nin temel nitelikleri nelerdir?\n<|assistant|>\n"

# Tokenize et ve modele ver
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Yanıt üret
output = model.generate(
    **inputs,
    max_new_tokens=200,
    do_sample=True,
    temperature=0.7,
    top_p=0.9
)

# Çıktıyı çözümle ve göster
print(tokenizer.decode(output[0], skip_special_tokens=True))
