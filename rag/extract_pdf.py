from pypdf import PdfReader

reader = PdfReader("rag_testing.pdf")
text = " ".join(page.extract_text() for page in reader.pages)

with open("data/raw/mypaper.txt", "w") as f:
    f.write(text)
print("extracted PDF -> data/raw/mypaper.txt")
