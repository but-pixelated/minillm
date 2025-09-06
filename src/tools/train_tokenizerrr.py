import sentencepiece as spm
import os

INP = "data/clean/mypaper_fixed.txt"
OUT_PREFIX = "data/spm_bpe_16k"
VOCAB_SIZE = 10000

os.makedirs("data", exist_ok=True)

spm.SentencePieceTrainer.Train(
    input=INP,
    model_prefix=OUT_PREFIX,
    vocab_size=VOCAB_SIZE,
    model_type="bpe",
    character_coverage=0.9995,
    train_extremely_large_corpus=True
)

print(f"[ok] trained tokenizer -> {OUT_PREFIX}.model / {OUT_PREFIX}.vocab")
