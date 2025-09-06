import os, re, unicodedata, glob

RAW_DIR = "data/raw"
OUT_DIR = "data/clean"
os.makedirs(OUT_DIR, exist_ok=True)

files = sorted(glob.glob(os.path.join(RAW_DIR, "*.txt")))
print("[*] found", len(files), "raw files")

out_path = os.path.join(OUT_DIR, "corpus.txt")
with open(out_path, "w", encoding="utf-8") as w:
    for f in files:
        with open(f, "r", encoding="utf-8", errors="ignore") as r:
            for line in r:
                line = unicodedata.normalize("NFKC", line)
                line = re.sub(r"\s+", " ", line).strip()
                if len(line) > 0:
                    w.write(line + "\n")

print("[ok] wrote clean corpus ->", out_path)
