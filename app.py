import torch
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict
import secrets
import uvicorn

# Import your model and tokenizer
from main import SimpleGPTClassifier, tokenizer, classify_text  # Adjust import paths

# Initialize FastAPI app
my_model = FastAPI()

# API Key for authentication
API_KEY = secrets.token_hex(16)  # Generates a random 32-character API key
AUTHORIZED_KEYS = {API_KEY}  # Store authorized API keys
print(f"This is your initial API key: {API_KEY}")

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleGPTClassifier(vocab_size=len(tokenizer.word2idx)).to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))  # Load trained model weights
model.eval()


class TextRequest(BaseModel):
    text: str


def verify_api_key(api_key: str):
    if api_key not in AUTHORIZED_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return api_key


@my_model.post("/predict", response_model=Dict[str, str])
async def predict(request: TextRequest, api_key: str):
    print(f"Received API Key: {api_key}")  # Log API key
    print(f"Received Request Body: {request.dict()}")  # Log request body
    
    if api_key not in AUTHORIZED_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    
    prediction = classify_text(request.text)
    return {"text": request.text, "prediction": prediction}

@my_model.get("/generate_key")
async def generate_key():
    new_key = secrets.token_hex(16)
    AUTHORIZED_KEYS.add(new_key)
    return {"api_key": new_key}

if __name__ == "__main__":
    uvicorn.run(my_model, host="127.0.0.1", port=8000)