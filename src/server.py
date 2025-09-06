import torch, sentencepiece as spm
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles 
from train_minigpt import GPT, device


sp = spm.SentencePieceProcessor(model_file="data/spm_bpe_16k.model")
ckpt = torch.load("out-minillm/best.pt", map_location=device)
config = ckpt["config"]

model = GPT().to(device)
model.load_state_dict(ckpt["model"])
model.eval()

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


app.mount("/web", StaticFiles(directory="web"), name="web")

@app.get("/")
def root():
    return FileResponse("web/index.html")


@app.post("/chat")
async def chat(req: Request):
    body = await req.json()
    prompt = body.get("prompt","")
    ids = sp.encode(prompt, out_type=int)
    x = torch.tensor([ids], device=device)
    for _ in range(100):
        x_cond = x[:, -config["block_size"]:]
        with torch.no_grad():
            logits,_ = model(x_cond)
            logits = logits[:, -1, :] / 0.9
            v,_ = torch.topk(logits, k=50)
            logits[logits < v[:,[-1]]] = -float("inf")
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs,1)
        x = torch.cat([x,next_id], dim=1)
    reply = sp.decode(x[0].tolist())
    return {"reply": reply}
 