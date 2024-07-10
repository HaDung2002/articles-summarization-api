from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import nest_asyncio
from pyngrok import ngrok
from pydantic import BaseModel
from typing import List, Dict
import os
from llama_cpp import Llama

PROJECT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)))
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# Load Mistral model

model_name = "mistral-7b-instruct-v0.2"
model_path = os.path.join(PROJECT_PATH, model_name+".Q4_K_M.gguf")
model = Llama(
    model_path=model_path,
    n_ctx = 2048,
    n_gpu_layers=35,
)

class Message(BaseModel):
    role: str
    content: str

class Request(BaseModel):
    messages: List[Message]
    model: str
    max_tokens: int
    temperature: float

@app.post("/chat/summary")
async def create_chat_summary(request: Request):
    if request.model != model_name:
        raise HTTPException(status_code=400, detail="Model not supported.")
    
    # Extract the latest message from user
    user_input = request.messages[-1].content

    output = model(
        user_input, # Prompt
        max_tokens=request.max_tokens,
        temperature=request.temperature,
    )
    summary = output['choices'][0]['text']

    return {
        "choices": [{"message": {"role": "assistant", "content": summary}}]
    }

if __name__ == "__main__":
    ngrok_tunnel = ngrok.connect(8000)
    print('Public URL:', ngrok_tunnel.public_url)
    nest_asyncio.apply()
    uvicorn.run(app, port=8000)