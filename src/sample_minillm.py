import torch, sentencepiece as spm
from train_minigpt import GPT, device

sp = spm.SentencePieceProcessor(model_file="data/spm_bpe_16k.model")


ckpt = torch.load("out-minillm/best.pt", map_location=device)
config = ckpt["config"]


model = GPT().to(device)
model.load_state_dict(ckpt["model"])
model.eval()

def generate(prompt, max_new=100, temperature=0.7, top_k=50):
    ids = sp.encode(prompt, out_type=int)
    x = torch.tensor([ids], device=device)
    for _ in range(max_new):
        x_cond = x[:, -config["block_size"]:]
        with torch.no_grad():
            logits,_ = model(x_cond)
            logits = logits[:, -1, :] / max(temperature,1e-6)
            if top_k:
                v,_ = torch.topk(logits, k=top_k)
                logits[logits < v[:,[-1]]] = -float("inf")
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs,1)
        x = torch.cat([x,next_id], dim=1)
    return sp.decode(x[0].tolist())

print(generate("how are you doing", 150))
