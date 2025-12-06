# Step1: Setup FastAPI backend
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ollama import Client
import ollama
from chromadb import Client as ChromaClient  # or your chroma client import
import os
from dotenv import load_dotenv

from Vector_without_db import retrieve_context
load_dotenv()




# OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY")
# OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "https://ollama.com")


# if not OLLAMA_API_KEY:
#     raise RuntimeError("Set OLLAMA_API_KEY environment variable")

# ollama_client = Client(
#     host=OLLAMA_HOST,
#     headers={"Authorization": f"Bearer {OLLAMA_API_KEY}"}
# )



#from ai_agent import graph, SYSTEM_PROMPT, parse_response

app = FastAPI()

# Step2: Receive and validate request from Frontend
class Query(BaseModel):
    message: str


@app.post("/ask")
async def ask(query: Query):
    try:
        # 1️⃣ Get relevant context from your document
        context_text = retrieve_context(query.message)

        # 2️⃣ Build messages with context included
        # messages = [
        #     {"role": "system", "content": (
        #             "You are an intelligent assistant. "
        #             "Use the below context to answer accurately. "
        #             "If the context is not helpful, answer normally.\n\n"
        #             f"### CONTEXT ###\n{context_text}\n### END CONTEXT ###"
        #         )},
        #     {"role": "user", "content": query.message}
        # ]

        messages = [
            {"role": "system", "content": (
                    "Use ONLY the given context. No external knowledge.If the context is not helpful, answer normally.\n\n"
                    f"### CONTEXT ###\n{context_text}\n### END CONTEXT ###"
                )},
            {"role": "user", "content": query.message}
        ]
        # 3️⃣ Call model
        resp = ollama.chat(model="qwen3-coder:480b-cloud", messages=messages, stream=False)
        
        # DEBUG: see the actual response
        # print(resp)  # Check your terminal to see the structure
        # answer = resp.message.content

        # 4️⃣ Extract answer safely
        if hasattr(resp, "message"):  
            # For new Ollama Python SDK (resp.message.content)
            answer = resp.message.content

        elif isinstance(resp, dict):
            if "message" in resp:
                answer = resp["message"]["content"]
            elif "output" in resp:
                answer = resp["output"]
            elif "choices" in resp:
                answer = resp["choices"][0]["message"]["content"]
            else:
                answer = str(resp)
        else:
            answer = str(resp)


        # TEMP: return raw resp to frontend
        return {"response": answer, "tool_called": None}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



    # response="this is response from backend"
    # return response

#actual
# async def ask(query: Query):
#     inputs = {"messages": [("system", SYSTEM_PROMPT), ("user", query.message)]}
#     #inputs = {"messages": [("user", query.message)]}
#     stream = graph.stream(inputs, stream_mode="updates")
#     tool_called_name, final_response = parse_response(stream)

#     # Step3: Send response to the frontend
#     return {"response": final_response,
#             "tool_called": tool_called_name}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)



