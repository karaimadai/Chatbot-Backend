# ============================================
# FASTAPI BACKEND — READY FOR RENDER + OLLAMA CLOUD
# ============================================

import os
import uvicorn
import ollama
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

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
    print("❌ ERROR: OLLAMA_API_KEY is missing. Cloud requests will fail.")

# --------------------------------------------
# Initialize Cloud Ollama Client (compatible mode)
# --------------------------------------------
client = ollama.Client(
    host=OLLAMA_HOST,
    headers={ 
        "Authorization": f"Bearer {OLLAMA_API_KEY}"
    }
)

print("🌐 Ollama Cloud Client Initialized")
print("🔧 HOST:", OLLAMA_HOST)
print("🔑 API Key Present:", "YES" if OLLAMA_API_KEY else "NO")
print("🤖 Using Model:", MODEL_PROVIDER)

# --------------------------------------------
# FastAPI App Setup
# --------------------------------------------
app = FastAPI()

# Enable CORS (WordPress Safe)
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
# HEALTH CHECK ENDPOINT
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

        # 1. Retrieve context with RAG
        context_text = retrieve_context(query.message)
        print("📚 RAG Context Length:", len(context_text))

        # 2. Build chat messages
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
        print("☁️ Calling Ollama Cloud model:", MODEL_PROVIDER)

        resp = client.chat(
            model=MODEL_PROVIDER,
            messages=messages,
            options={"temperature": 0.7}
        )

        # 4. Extract answer safely
        answer = None

        if isinstance(resp, dict):
            # Standard Ollama Cloud schema
            if "message" in resp:
                answer = resp["message"].get("content")

            # Fallbacks
            if not answer:
                answer = (
                    resp.get("output")
                    or resp.get("content")
                    or (resp.get("choices", [{}])[0]
                        .get("message", {})
                        .get("content"))
                )

        if not answer:
            answer = "Sorry, I couldn't generate a response."

        print("✅ Final Answer:", answer[:120], "...")

        return {"response": answer}

    except Exception as e:
        print("🔥 ERROR IN /ask:", str(e))
        raise HTTPException(status_code=500, detail=str(e))

# --------------------------------------------
# LOCAL DEV ENTRY POINT
# --------------------------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
