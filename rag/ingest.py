import faiss, pickle
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader


reader = PdfReader("rag_testing.pdf")
text = " ".join(page.extract_text() for page in reader.pages)


def chunk_text(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

chunks = chunk_text(text)


model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks)


index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

with open("rag_store.pkl", "wb") as f:
    pickle.dump((chunks, index, model), f)

print(f"stored {len(chunks)} chunks in vector DB")
