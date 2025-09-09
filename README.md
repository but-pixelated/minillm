<p align="center">
  <img src="assets/demo.png" alt="minillm demo" width="800"/>
</p>

# minillm — local llm + rag demo

this repo shows how to **train your own mini-llm**, add a **retrieval-augmented generation (rag) pipeline**, and serve a **pretrained hf model** with a simple web ui.  
tested on macbook pro m1. lightweight + educational.

---

## features
- **train from scratch** → tiny gpt-like model on your own dataset  
- **rag** → index + search docs with embeddings + faiss (vector search)  
- **pretrained model** → integrate huggingface models (`CraneAILabs/ganda-gemma-1b` by default)  
- **web ui** → minimal chat-like interface (`web/index.html`)  
- **fastapi server** → backend serving, easy to extend  

---

## repo structure
minillm/
├── src/ # training + serving code <br>
│ ├── server_hf.py # fastapi server for pretrained model <br>
│ ├── server.py # (optional) server for scratch trained model <br>
│ ├── train_minigpt.py # training loop for mini-llm <br>
│ ├── sample_minillm.py <br>
│ ├── model.py # gpt like model definition <br>
│ └── tools/ # helpers: tokenizer training, binify, corpus downloader <br>
│  <br>
├── rag/ # retrieval augmented generation(RAG) pipeline <br>
│ ├── ingest.py # chunk + embed documents <br>
│ ├── build_index.py # build faiss/annoy index <br>
│ ├── search.py # query top k passages <br>
│ └── index/ # generated vector indexes (ignored in git) <br>
│ <br>
├── web/ # minimal html/js ui <br>
│ └── index.html <br>
│ <br>
├── tokenizer/ # optional tokenizer artifacts <br>
│ ├── spm_bpe_16k.model <br>
│ └── spm_bpe_16k.vocab <br>
│ <br>
├── checkpoints/ # training checkpoints (ignored in git) <br>
│ └── out-minillm/best.pt <br>
│ <br>
├── pretrained/ # pointers for hf pretrained models <br>
│ └── README_PRETRAINED.md <br>
│ <br>
├── scripts/ # helpers for downloads + setup <br>
│ └── download_weights.sh <br>
│ <br>
├── README.md <br>
├── README_RAG.md <br>
├── requirements.txt <br>
└── .gitignore <br>


---

## quickstart

### 1. setup environment
```bash
python3 -m venv ~/.venvs/minillm
source ~/.venvs/minillm/bin/activate
pip install -r requirements.txt
2. run the pretrained server

uvicorn src.server_hf:app --reload --port 8000
open browser → http://127.0.0.1:8000
you’ll see the chat ui running locally.

3. run rag indexing
see README_RAG.md for details.

4. train from scratch

python src/train_minigpt.py
checkpoints saved to checkpoints/out-minillm/.

handling large files
model checkpoints (.pt, .bin), rag indexes (.faiss, .pkl) → not committed (see .gitignore)

download them using scripts/download_weights.sh

pretrained model instructions in README_PRETRAINED.md


license & credits

training code → inspired by nanoGPT & HF examples
pretrained → CraneAILabs/ganda-gemma-1b
rag → faiss + sentence-transformers
ui → minimal html/css/js
licensed under mit. datasets/models used must follow their respective licenses.
