import requests

API_KEY = "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"

r = requests.post(
    "https://integrate.api.nvidia.com/v1/chat/completions",
    headers={"Authorization": f"Bearer {API_KEY}"},
    json={
        "model": "meta/llama-3.1-8b-instruct",
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0
    }
)

print(r.status_code)
print(r.text)