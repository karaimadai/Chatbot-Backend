# ============================================
#   FASTAPI + OLLAMA CLOUD BACKEND FOR RENDER
# ============================================

import os
import uvicorn
import ollama
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from Vector_without_db import retrieve_context

# --------------------------------------------
# Load .env for LOCAL DEV — ignored on Render
# --------------------------------------------
load_dotenv()

# --------------------------------------------
# Load ENV variables (Render injects these)
# --------------------------------------------
MODEL_PROVIDER = os.environ.get("MODEL_PROVIDER", "qwen3-coder:480b-cloud")
OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "https://api.ollama.com")

# FORCE environment variables (important)
os.environ["OLLAMA_API_KEY"] = OLLAMA_API_KEY if OLLAMA_API_KEY else ""
os.environ["OLLAMA_HOST"] = OLLAMA_HOST

print("🔧 Using Model:", MODEL_PROVIDER)
print("🔧 OLLAMA_HOST:", os.environ.get("OLLAMA_HOST"))
print("🔧 API Key Present:", "YES" if OLLAMA_API_KEY else "NO")

# --------------------------------------------
# Initialize FastAPI
# --------------------------------------------
app = FastAPI()

# CORS (works for any frontend including WordPress)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------
# Data Model
# --------------------------------------------
class Query(BaseModel):
    message: str

# --------------------------------------------
# HEALTH CHECK ENDPOINT
# --------------------------------------------
@app.get("/")
def health():
    return {
        "status": "ok",
        "service": "AI Chatbot Backend",
        "model_provider": MODEL_PROVIDER,
        "ollama_host": os.environ.get("OLLAMA_HOST"),
    }

# --------------------------------------------
# MAIN CHAT ENDPOINT
# --------------------------------------------
@app.post("/ask")
async def ask(query: Query):
    try:
        print("📩 Incoming user message:", query.message)

        # 1. Retrieve context using RAG
        context_text = retrieve_context(query.message)
        print("📚 Retrieved context length:", len(context_text))

        # 2. Build prompt
        messages = [
            {
                "role": "system",
                "content": (
                    "Use ONLY the given context. If the context is not helpful, "
                    "answer normally.\n\n"
                    f"### CONTEXT ###\n{context_text}\n### END CONTEXT ###"
                ),
            },
            {"role": "user", "content": query.message},
        ]

        # 3. Call Ollama Cloud
        print("🤖 Calling Ollama Cloud model:", MODEL_PROVIDER)

        resp = ollama.chat(
            model=MODEL_PROVIDER,
            messages=messages,
            stream=False,
        )

        # 4. Safely extract response
        answer = None

        if isinstance(resp, dict):
            msg = resp.get("message", {})
            answer = msg.get("content")

            # Fallbacks
            if not answer:
                if "output" in resp:
                    answer = resp["output"]
                elif "content" in resp:
                    answer = resp["content"]
                elif "choices" in resp:
                    answer = resp["choices"][0]["message"]["content"]

        if not answer:
            answer = "Sorry, the model did not return any response."

        print("✅ Final answer:", answer[:80], "...")

        return {"response": answer}

    except Exception as e:
        print("🔥 ERROR IN /ask:", str(e))
        raise HTTPException(status_code=500, detail=str(e))


# --------------------------------------------
# LOCAL DEV
# --------------------------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
