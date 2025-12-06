# Step1: Setup FastAPI backend
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import ollama
from dotenv import load_dotenv

from Vector_without_db import retrieve_context

# Load .env locally (ignored on Render)
load_dotenv()

# -----------------------------
# Load Render Environment Vars
# -----------------------------
MODEL_PROVIDER = os.environ.get("MODEL_PROVIDER", "qwen3-coder:480b-cloud")
OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY")
OLLAMA_HOST = "https://api.ollama.com"     # Required for Ollama Cloud

# Force API Key into environment for SDK
if OLLAMA_API_KEY:
    os.environ["OLLAMA_API_KEY"] = OLLAMA_API_KEY
else:
    print("WARNING: OLLAMA_API_KEY is missing! The model will return 500 errors.")

# -----------------------------
# Create FastAPI App + CORS
# -----------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # Allow WordPress & everywhere
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request Model
class Query(BaseModel):
    message: str

# -----------------------------
# API ROUTE /ask
# -----------------------------
@app.post("/ask")
async def ask(query: Query):
    try:
        # 1. Retrieve RAG Context
        context_text = retrieve_context(query.message)

        # 2. Build messages for LLM
        messages = [
            {
                "role": "system",
                "content": (
                    "Use ONLY the given context. No external knowledge.\n"
                    "If the context is not helpful, answer normally.\n\n"
                    f"### CONTEXT ###\n{context_text}\n### END CONTEXT ###"
                ),
            },
            {"role": "user", "content": query.message},
        ]

        # 3. Call Ollama Cloud
        resp = ollama.chat(
            model=MODEL_PROVIDER,
            messages=messages,
            stream=False,
            options={"temperature": 0.7}
        )

        # 4. Extract model output safely
        answer = None

        if isinstance(resp, dict):
            if "message" in resp and "content" in resp["message"]:
                answer = resp["message"]["content"]
            elif "output" in resp:
                answer = resp["output"]
            elif "content" in resp:
                answer = resp["content"]
            elif "choices" in resp:
                answer = resp["choices"][0]["message"]["content"]

        if not answer:
            answer = "Sorry, I could not generate a response."

        # Send JSON response
        return {"response": answer}

    except Exception as e:
        # Print error (shows in Render logs)
        print("🔥 ERROR IN /ask:", str(e))
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------
# LOCAL RUN
# -----------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
