import os
from dotenv import load_dotenv
import ollama

# Load environment variables
load_dotenv()

response = ollama.chat(
    model='qwen3-coder:480b-cloud',    # choose a model that exists locally
    messages=[
        {
            'role': 'user',
            'content': 'why is sky blue?',
        }
    ]
)

print(response['message']['content'])
