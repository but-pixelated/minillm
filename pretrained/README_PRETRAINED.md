# `README_PRETRAINED.md`
```markdown
# pretrained model usage

this repo supports running hf pretrained models instead of scratch-trained weights.  
default: [`CraneAILabs/ganda-gemma-1b`](https://huggingface.co/CraneAILabs/ganda-gemma-1b)

---

## ⚡ quickstart
```bash
uvicorn src.server_hf:app --reload --port 8000
open → http://127.0.0.1:8000

🔑 huggingface auth (if needed)
some models are gated. to use:

bash

pip install huggingface_hub
huggingface-cli login
or set env var:

bash

export HUGGINGFACE_HUB_TOKEN="hf_..."
📦 model caching
when run, hf downloads weights to local cache:

mac: ~/Library/Caches/huggingface/hub

linux: ~/.cache/huggingface/hub


📥 custom checkpoints
if you trained your own model (checkpoints/out-minillm/best.pt), place it here:

bash

checkpoints/out-minillm/best.pt
then run src/server.py instead of server_hf.py.

📝 credits
gemma models: google / craneai labs

transformers library: huggingface

---

these three give you:  
- `README.md` → full project overview  
- `README_RAG.md` → rag-specific instructions  
- `README_PRETRAINED.md` → pretrained hf model usage  


