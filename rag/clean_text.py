import re

with open("data/raw/mypaper.txt", "r") as f:
    text = f.read()

text = re.sub(r"\[\d+\]", "", text)      
text = re.sub(r"\(\d{4}\)", "", text)     
text = re.sub(r"\s+", " ", text)         

with open("data/clean/mypaper.txt", "w") as f:
    f.write(text)

print("✅ cleaned -> data/clean/mypaper.txt")
