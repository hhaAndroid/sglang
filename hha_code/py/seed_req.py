#!/usr/bin/env python3

import requests

url = "http://10.102.249.52:41145/v1/chat/completions"

data = {
    "model": "Qwen/Qwen3.5-35B-A3B", # Qwen/Qwen3.5-35B-A3B / XiaomiMiMo/MiMo-7B-RL
    "messages": [{"role": "user", "content": "Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\nDetermine the smallest prime $p$ such that $2018!$ is divisible by $p^3$, but not divisible by $p^4$.\n\nRemember to put your answer on its own line after \"Answer:\"."}],
}

response = requests.post(url, json=data)
print(response.json())
