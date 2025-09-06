import torch, pickle
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sentence_transformers import SentenceTransformer
from train_minigpt import GPT, device
import sentencepiece as spm
import faiss

# load GPT + tokenizer
sp = spm.SentencePieceProcessor(model_file="data/spm_bpe_16k.model")
ckpt = torch.load("out-minillm/best.pt", map_location=device)
config = ckpt["config"]
model = GPT().to(device)
model.load_state_dict(ckpt["model"])
model.eval()

# load RAG store
with open("rag_store.pkl", "rb") as f:
    chunks, index, embedder = pickle.load(f)

# fastapi setup
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

app.mount("/web", StaticFiles(directory="web"), name="web")

@app.get("/")
def root(): 
    return FileResponse("web/index.html")

@app.post("/chat")
async def chat(req: Request):
    body = await req.json()
    query = body.get("prompt","")

    # embed query + retrieve chunks
    q_emb = embedder.encode([query])
    D,I = index.search(q_emb, k=2)
    context = "\n".join(chunks[i][:400] for i in I[0])

    # augment prompt
    final_prompt = f"Answer based on the following text:\n{context}\n\nQuestion: {query}\nAnswer:"

    # encode to tokens
    ids = sp.encode(final_prompt, out_type=int)
    x = torch.tensor([ids], device=device)

    # 🔥 generation loop (properly indented inside chat function)
    temperature = 0.6
    top_k = 30
    for _ in range(100):
        x_cond = x[:, -config["block_size"]:]
        with torch.no_grad():
            logits,_ = model(x_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            v,_ = torch.topk(logits, k=top_k)
            logits[logits < v[:,[-1]]] = -float("inf")
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs,1)
        x = torch.cat([x,next_id], dim=1)

    reply = sp.decode(x[0].tolist())
    return {"reply": reply}
