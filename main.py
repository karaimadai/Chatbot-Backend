from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from ollama import Client
import os
from dotenv import load_dotenv
from Vector_without_db import retrieve_context

load_dotenv()

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

# ------------------------
# Ollama Cloud Client Fix
# ------------------------
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")

if not OLLAMA_API_KEY:
    raise RuntimeError("❌ OLLAMA_API_KEY missing in Render environment")

ollama_client = Client(
    host="https://api.ollama.ai",
    headers={"Authorization": f"Bearer {OLLAMA_API_KEY}"}
)

@app.post("/ask")
async def ask(query: Query):
    try:
        context_text = retrieve_context(query.message)

        messages = [
            {"role": "system", "content": (
                "Use ONLY the given context. If not helpful, answer normally.\n\n"
                f"### CONTEXT ###\n{context_text}\n### END CONTEXT ###"
            )},
            {"role": "user", "content": query.message}
        ]

        resp = ollama_client.chat(
            model="qwen2.5:7b",  # ← Use a valid cloud model
            messages=messages,
            stream=False
        )

        answer = resp["message"]["content"]

        return {"response": answer}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
