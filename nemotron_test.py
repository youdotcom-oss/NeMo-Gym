import os

import requests

response = requests.post(
    "https://api.inference.wandb.ai/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {os.getenv('WEIGHTS_AND_BIASES_INFERENCE_KEY')}",
        "OpenAI-Project": "eoyou/nemotron-eval"
    },
    json={
        "model": "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Tell me a joke."},
        ],
    },
)
response.raise_for_status()

print(response.json()["choices"][0]["message"]["content"])