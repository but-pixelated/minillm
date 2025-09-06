import sentencepiece as spm

with open("data/clean/mypaper.txt", "r", encoding="utf-8", errors="ignore") as f:
    text = f.read()


import re
sentences = re.split(r'(?<=[.!?]) +', text)
fixed_lines = []
for s in sentences:
    s = s.strip()
    if len(s) > 0:
        while len(s) > 5000:
            fixed_lines.append(s[:5000])
            s = s[5000:]
        fixed_lines.append(s)

with open("data/clean/mypaper_fixed.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(fixed_lines))

print(f"wrote {len(fixed_lines)} sentences to data/clean/mypaper_fixed.txt")


spm.SentencePieceTrainer.Train(
    input="data/clean/mypaper_fixed.txt",
    model_prefix="data/spm_bpe_16k",
    vocab_size=4000,
    model_type="bpe",
    character_coverage=1.0,
    max_sentence_length=10000
)

print("tokenizer trained -> data/spm_bpe_16k.model, data/spm_bpe_16k.vocab")
