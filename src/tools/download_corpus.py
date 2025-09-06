import os, re
from datasets import load_dataset
from tqdm import tqdm

OUTDIR = "data/raw"
os.makedirs(OUTDIR, exist_ok=True)

def sanitize(txt: str) -> str:
    if not isinstance(txt, str):
        return ""
    txt = txt.replace("\r", " ")
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt

def write_stream(name, ds_iter, field="text", target_gb=1.0):
    target_bytes = int(target_gb * (1024**3))
    out_path = os.path.join(OUTDIR, f"{name}.txt")
    written = 0
    n = 0
    with open(out_path, "w", encoding="utf-8") as w, tqdm(unit="B", unit_scale=True, desc=f"{name}") as pbar:
        for ex in ds_iter:
            txt = sanitize(ex.get(field, ""))
            if not txt:
                continue
            b = (txt + "\n").encode("utf-8")
            w.write(txt + "\n")
            written += len(b)
            n += 1
            pbar.update(len(b))
            if written >= target_bytes:
                break
    print(f"[ok] {name}: wrote ~{written/1024/1024:.1f} MB ({n} samples) -> {out_path}")

def main():

    ds = load_dataset(
        "wikimedia/wikipedia", "20231101.en",
        streaming=True, split="train",
        trust_remote_code=True
    )
    write_stream("wikipedia_en", ds, field="text", target_gb=1.0)

if __name__ == "__main__":
    main()
