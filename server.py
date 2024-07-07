from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import nest_asyncio
from pyngrok import ngrok
from pydantic import BaseModel
from typing import List, Dict
import os
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_inference.model import Transformer
from mistral_inference.generate import generate

PROJECT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)))
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# Load Mistral model and tokenizer
model_name = "mistral-7B-Instruct-v0.3"
mistral_models_path = os.path.join(PROJECT_PATH, model_name)
tokenizer = MistralTokenizer.v3()
model = Transformer.from_folder(mistral_models_path)

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

    completion_request = ChatCompletionRequest(messages=[UserMessage(content=user_input)])
    inputs = tokenizer.encode_chat_completion(completion_request).tokens
    outputs, _ = generate([inputs], model,
                          max_tokens=request.max_tokens,
                          temperature=request.temperature,
                          eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id)
    summary = tokenizer.decode(outputs[0])

    return {
        "choices": [{"message": {"role": "assistant", "content": summary}}]
    }

if __name__ == "__main__":
    ngrok_tunnel = ngrok.connect(8000)
    print('Public URL:', ngrok_tunnel.public_url)
    nest_asyncio.apply()
    uvicorn.run(app, port=8000)