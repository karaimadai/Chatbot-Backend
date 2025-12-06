import os
import re

# ------------------------------------------------------------
# 3️⃣ Load your document
# ------------------------------------------------------------
#file_path = "C:/Users/sridh/OneDrive/Desktop/LLM Projects/Handson Chatbot Creation/Sample text Document.txt"
file_path = "C:/Users/sridh/OneDrive/Desktop/LLM Projects/Handson Chatbot Creation/ekom.txt"

if not os.path.exists(file_path):
    raise FileNotFoundError(f"❌ File not found: {file_path}")

with open(file_path, "r", encoding="utf-8") as f:
    text_data = f.read()

# ------------------------------------------------------------
# 4️⃣ Split text into small chunks for simple retrieval
# ------------------------------------------------------------
def split_text(text, chunk_size=400, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

chunks = split_text(text_data)
print(f"✅ Split document into {len(chunks)} chunks.")

# ------------------------------------------------------------
# 5️⃣ Simple keyword-based context retriever
# ------------------------------------------------------------

def retrieve_context(query, top_k=3):
    # Score each chunk by number of overlapping words
    query_words = set(re.findall(r"\w+", query.lower()))
    scores = []
    for i, chunk in enumerate(chunks):
        chunk_words = set(re.findall(r"\w+", chunk.lower()))
        overlap = len(query_words & chunk_words)
        scores.append((overlap, i))
    scores.sort(reverse=True)
    top_chunks = [chunks[i] for _, i in scores[:top_k] if _ > 0]
    return "\n".join(top_chunks) if top_chunks else "No relevant context found."
