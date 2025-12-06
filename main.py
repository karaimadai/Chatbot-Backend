# ============================================
# FASTAPI BACKEND (100% CLOUD READY for Render)
# Uses new Ollama Cloud Client (ollama>=0.4.4)
# ============================================

import os
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import ollama

from Vector_without_db import retrieve_context

# Load .env locally (ignored on Render)
load_dotenv()

# --------------------------------------------
# Load Render Environment Variables
# --------------------------------------------
MODEL_PROVIDER = os.environ.get("MODEL_PROVIDER", "qwen3-coder:480b-cloud")
OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "https://api.ollama.com")

if not OLLAMA_API_KEY:
    print("❌ OLLAMA_API_KEY NOT FOUND — Cloud requests will fail!")

# --------------------------------------------
# Initialize Cloud Ollama Client (New SDK)
# --------------------------------------------
client = ollama.Client(
    host=OLLAMA_HOST,
    api_key=OLLAMA_API_KEY
)

print("🌐 Ollama Cloud Client Initialized")
print("🔧 HOST:", OLLAMA_HOST)
print("🔑 API Key Present:", "YES" if OLLAMA_API_KEY else "NO")
print("🤖 Using Model:", MODEL_PROVIDER)

# --------------------------------------------
# FastAPI App Setup
# --------------------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    message: str

# --------------------------------------------
# HEALTH CHECK
# --------------------------------------------
@app.get("/")
def health():
    return {
        "status": "ok",
        "model_provider": MODEL_PROVIDER,
        "ollama_host": OLLAMA_HOST,
    }

# --------------------------------------------
# MAIN CHAT ENDPOINT
# --------------------------------------------
@app.post("/ask")
async def ask(query: Query):
    try:
        print("📩 User Query:", query.message)

        # 1. Retrieve RAG context
        context_text = retrieve_context(query.message)
        print("📚 Retrieved Context Length:", len(context_text))

        # 2. Build messages
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

        # 3. Call Ollama Cloud (explicit client)
        print("☁️ Calling Ollama Cloud model:", MODEL_PROVIDER)

        resp = client.chat(
            model=MODEL_PROVIDER,
            messages=messages,
            options={"temperature": 0.7}
        )

        # 4. Extract reply
        answer = None
        if isinstance(resp, dict):
            msg = resp.get("message", {})
            answer = msg.get("content")

            if not answer:
                answer = resp.get("output") or resp.get("content")

        if not answer:
            answer = "Sorry, I could not generate a response."

        print("✅ Final Answer:", answer[:100], "...")

        return {"response": answer}

    except Exception as e:
        print("🔥 ERROR in /ask:", str(e))
        raise HTTPException(status_code=500, detail=str(e))


# --------------------------------------------
# LOCAL RUN
# --------------------------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
