from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi import FastAPI
from pydantic import BaseModel

import time
import re

# --- Model Yükleme ---
model_id = "LiquidAI/LFM2-1.2B"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype="bfloat16",
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# --- FastAPI Uygulaması Başlat ---
app = FastAPI()

# --- İstekten Gelen Veriyi Modellemek İçin Pydantic Sınıfı ---
class ChatSystem(BaseModel):
    prompt: str  # Kullanıcının sohbet için gönderdiği metin

# --- Chatbot API Endpoint'i ---
@app.post("/chat/")
def chatting(prompt: ChatSystem):
    # İstek işleme başlangıç zamanı (performans ölçümü için)
    start_time = time.time()
    
    # Kullanıcı mesajını model için uygun formatta tokenize et
    input_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt.prompt}],
        add_generation_prompt=True,
        return_tensors="pt",
        tokenize=True,
    ).to(model.device)

    # Modelden yanıt üret
    output = model.generate(
        input_ids,
        do_sample=True,
        temperature=0.3,
        min_p=0.15,
        repetition_penalty=1.05,
        max_new_tokens=512,
    )

    # Model çıktısını string haline çevir
    full_text = tokenizer.decode(output[0], skip_special_tokens=False)

    # Modelin asistan cevabını ayıkla (sadece assistant mesajını al)
    matches = re.findall(r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>", full_text, re.DOTALL)
    assistant_reply = matches[0].strip() if matches else "Cevap bulunamadı."

    # İstek işleme bitiş zamanı
    end_time = time.time()
    elapsed = end_time - start_time
    
    # Yanıt ve işlem süresini JSON olarak geri döndür
    return {"response": assistant_reply, "elapsed_time": elapsed}
