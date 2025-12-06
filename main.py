# =====================================================
# FASTAPI + OLLAMA CLOUD VIA HTTPS (NOT OLLAMA SDK)
# =====================================================

import os
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from Vector_without_db import retrieve_context

# Load env (local only)
load_dotenv()

# Environment variables from Render
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "qwen3-coder:480b-cloud")

if not OLLAMA_API_KEY:
    raise RuntimeError("OLLAMA_API_KEY is missing in Render environment variables")

# FastAPI setup
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

@app.get("/")
def health():
    return {"status": "ok", "model": MODEL_PROVIDER}

@app.post("/ask")
async def ask(query: Query):
    try:
        print("📩 User:", query.message)

        # Get context from your RAG function
        context_text = retrieve_context(query.message)
        print("📚 Context length:", len(context_text))

        # Build messages
        messages = [
            {
                "role": "system",
                "content": (
                    "Use ONLY the given context. If context is not useful, answer normally.\n\n"
                    f"### CONTEXT ###\n{context_text}\n### END CONTEXT ###"
                )
            },
            {"role": "user", "content": query.message}
        ]

        # Call Ollama Cloud
        print("☁️ Sending request to Cloud Ollama...")

        response = requests.post(
            "https://api.ollama.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OLLAMA_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL_PROVIDER,
                "messages": messages,
            },
            timeout=60
        )

        if response.status_code != 200:
            print("❌ Cloud Error:", response.text)
            raise HTTPException(status_code=500, detail=response.text)

        data = response.json()
        answer = data["choices"][0]["message"]["content"]

        print("✅ Reply:", answer[:100], "...")

        return {"response": answer}

    except Exception as e:
        print("🔥 ERROR /ask:", str(e))
        raise HTTPException(status_code=500, detail=str(e))
