import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Başlık
st.set_page_config(page_title="JustiBot - Hukuk Asistanı", layout="centered")
st.title("⚖️ JustiBot - Hukuk Asistanınız")
st.write("Eğitilmiş yerel modeli kullanarak hukuki sorularınıza yanıt verir.")

# Model ve tokenizer'ı önbelleğe alarak yükle
@st.cache_resource
def load_model():
    model_path = "tiny-lawbot-model"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    return tokenizer, model

tokenizer, model = load_model()

# Kullanıcıdan giriş al
user_input = st.text_area("🔍 Sorunuzu yazın:", height=100, placeholder="Örnek: Anayasa madde 2’ye göre Türkiye Cumhuriyeti’nin temel nitelikleri nelerdir?")

if st.button("🧠 Yanıtla"):
    if user_input.strip() == "":
        st.warning("Lütfen bir soru girin.")
    else:
        with st.spinner("Model yanıtlıyor..."):
            # Prompt formatı
            prompt = f"Prompt: {user_input}\nResponse:"

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id
                )

            response = tokenizer.decode(output[0], skip_special_tokens=True)
            # Cevabın yalnızca Response kısmını ayıklayalım
            if "Response:" in response:
                final_output = response.split("Response:")[-1].strip()
            else:
                final_output = response.strip()

            st.success("🗣️ Yanıt:")
            st.write(final_output)
