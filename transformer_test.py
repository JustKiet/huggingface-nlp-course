import torch
from transformers import pipeline
import os
from huggingface_hub import login
from dotenv import load_dotenv
from pprint import pprint

from typing import Dict, List

load_dotenv()

API_KEY = os.getenv("HUGGINGFACE_API_KEY") 

login(token=API_KEY)

model_id = "meta-llama/Llama-3.2-3B-Instruct"
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

messages = [
    {"role": "system", "content": "You are a typical chatbot, trying to assist the user in the best way possible."},
]

while True:
    user_input = input("User: ")
    
    if user_input.lower() == "exit":
        print("Shutting down.")
        break
    
    formatted_input = {
        "role": "user",
        "content": user_input,
    }
    
    messages.append(formatted_input)
    
    outputs = pipe(
        messages,
        max_new_tokens=256,
    )
    
    output_content = {
        "role": "assistant",
        "content": outputs[0]["generated_text"][-1]
    }
    
    messages.append(output_content)
    
    print("-----------------------------------------")
    [pprint(message) for message in messages]
    print("-----------------------------------------")
    print(outputs[0]["generated_text"][-1])