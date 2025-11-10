# main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import logging

# ================== تنظیمات لاگ‌گیری ==================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================== تنظیمات اولیه مدل ==================
MODEL_NAME = "nafisehNik/mt5-persian-summary"  # مدل فارسی برای خلاصه‌سازی :contentReference[oaicite:2]{index=2}
logger.info(f"Loading model {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
logger.info(f"Model loaded to device: {device}")

# ================== ساخت برنامه FastAPI ==================
app = FastAPI(
    title="Persian Text Summarization API",
    description="An API for summarizing Persian text using a HuggingFace model.",
    version="0.1.0"
)

# ================== مدل‌های داده برای ورودی/خروجی ==================
class TextIn(BaseModel):
    text: str
    max_input_length: int = 512        # حداکثر طول ورودی به توکن‌ها
    max_summary_length: int = 150      # حداکثر طول خروجی (خلاصه)
    num_beams: int = 4                 # تعداد پرتوها (beams) برای تولید
    length_penalty: float = 2.0        # جریمه طول (طول‌تر = کمتر امتیاز)
    repetition_penalty: float = 1.2    # جریمه تکرار
    early_stopping: bool = True        # آیا تولید زودتر متوقف شود؟

class SummaryOut(BaseModel):
    summary: str

# ================== تابع خلاصه‌سازی ==================
def make_summary(text: str,
                 max_input_length: int = 512,
                 max_summary_length: int = 150,
                 num_beams: int = 4,
                 length_penalty: float = 2.0,
                 repetition_penalty: float = 1.2,
                 early_stopping: bool = True
                 ) -> str:
    # توکنایز متن
    inputs = tokenizer(text,
                       return_tensors="pt",
                       truncation=True,
                       padding="longest",
                       max_length=max_input_length,
                       add_special_tokens=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    logger.info(f"Input length (tokens): {input_ids.shape}")

    # تولید خلاصه
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=max_summary_length,
        num_beams=num_beams,
        length_penalty=length_penalty,
        repetition_penalty=repetition_penalty,
        early_stopping=early_stopping,
        no_repeat_ngram_size=3,       # جلوگیری از تکرار n-gramها
        min_length=30,               # حداقل طول خلاصه
        do_sample=False,              # نمونه‌گیری خاموش
        use_cache=True
    )

    summary = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    logger.info(f"Generated summary length: {len(summary.split())} words")

    return summary

# ================== تعریف endpoint ==================
@app.post("/summarize", response_model=SummaryOut)
async def summarize_endpoint(input_data: TextIn):
    text = input_data.text
    if not text or text.strip() == "":
        raise HTTPException(status_code=400, detail="Input text must be non-empty.")
    try:
        summary = make_summary(
            text=text,
            max_input_length=input_data.max_input_length,
            max_summary_length=input_data.max_summary_length,
            num_beams=input_data.num_beams,
            length_penalty=input_data.length_penalty,
            repetition_penalty=input_data.repetition_penalty,
            early_stopping=input_data.early_stopping
        )
        return SummaryOut(summary=summary)
    except Exception as e:
        logger.error(f"Error during summarization: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error during summarization: {str(e)}")

# ================== نقطه شروع (اختیاری) ==================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
