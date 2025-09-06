import os
import torch
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_NAME = "CraneAILabs/ganda-gemma-1b"  
SYSTEM_PROMPT = (
    "you are a friendly, concise ai assistant. keep answers short (1-4 sentences) "
    "and conversational. do not invent fake conversation logs. if the user says "
    "something short (e.g. 'hello'), respond briefly and politely. use emojis sparingly."
)


GEN_KWARGS = {
    "max_new_tokens": 60,
    "temperature": 0.5,
    "top_k": 40,
    "do_sample": True,
    "pad_token_id": None,  
}


device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"[info] device: {device}")

torch_dtype = torch.float16 if device == "mps" else torch.float32

print(f"[info] loading tokenizer + model: {MODEL_NAME} (this may download files)...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, trust_remote_code=False)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

GEN_KWARGS["pad_token_id"] = tokenizer.eos_token_id

model = None
try:
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch_dtype)
    model.to(device)
except Exception as e:
    print("[warn] fp16 load failed or unsupported on this env:", e)
    print("[info] retrying with default dtype and device map...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.to(device)

model.eval()
print("[ok] model loaded and in eval mode")


app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/web", StaticFiles(directory="web"), name="web")

@app.get("/")
def root():
    return FileResponse("web/index.html")

@app.post("/chat")
async def chat(req: Request):
    body = await req.json()
    user_prompt = body.get("prompt", "").strip()
    if not user_prompt:
        return JSONResponse({"reply": "say something, i'm listening 👂"}, status_code=400)

    
    SYSTEM_PROMPT = (
        "you are a friendly, concise assistant. answer directly and briefly. "
        "do not produce or mimic 'user:'/'assistant:' transcripts. keep replies 1-3 sentences."
    )

    
    prompt = f"{SYSTEM_PROMPT}\n\n{user_prompt}\n\nassistant:"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    
    gen_kwargs = {
        "max_new_tokens": 50,
        "temperature": 0.4,
        "top_k": 30,
        "do_sample": True,
        "no_repeat_ngram_size": 3,
        "repetition_penalty": 1.05,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    with torch.no_grad():
        out_ids = model.generate(**inputs, **gen_kwargs)

    text = tokenizer.decode(out_ids[0], skip_special_tokens=True)

    
    if prompt in text:
        reply = text.split(prompt, 1)[1].strip()
    else:
        
        if text.startswith(user_prompt):
            reply = text[len(user_prompt):].strip()
        else:
            reply = text.strip()

    
    reply = reply.split("\n\n")[0].strip()
    if len(reply) > 600:
        reply = reply[:600].rsplit(" ", 1)[0] + "..."

    if not reply:
        reply = "hm, couldn't form a response — try rephrasing?"

    return {"reply": reply}

