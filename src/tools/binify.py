import sentencepiece as spm
import numpy as np
import os

INP = "data/clean/mypaper_fixed.txt"
MODEL = "data/spm_bpe_16k.model"
OUTDIR = "data/bin"
os.makedirs(OUTDIR, exist_ok=True)

sp = spm.SentencePieceProcessor(model_file=MODEL)

ids = []
with open(INP, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            ids.extend(sp.encode(line, out_type=int) + [sp.eos_id()])

arr = np.array(ids, dtype=np.uint16)  
out_path = os.path.join(OUTDIR, "corpus_u16.bin")
arr.tofile(out_path)

print(f"[ok] wrote {len(arr)} tokens -> {out_path}")
