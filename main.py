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

        # Get RAG context
        context_text = retrieve_context(query.message)
        print("📚 Context length:", len(context_text))

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

        print("☁️ Sending request to Cloud Ollama...")

        response = requests.post(
            "https://cloud.ollama.com/v1/chat/completions",
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

        print("📥 RAW RESPONSE STATUS:", response.status_code)
        print("📥 RAW RESPONSE BODY:", response.text[:500])

        if response.status_code != 200:
            return {"error": "cloud_error", "detail": response.text}

        data = response.json()

        # handle empty results
        try:
            answer = data["choices"][0]["message"]["content"]
        except:
            answer = "(Empty response from model)"

        print("✅ Final Answer:", answer[:200])

        # TEMP: return FULL CLOUD RESPONSE for debugging
        return {
            "response": answer,
            "raw": data
        }

    except Exception as e:
        print("🔥 ERROR /ask:", str(e))
        raise HTTPException(status_code=500, detail=str(e))
