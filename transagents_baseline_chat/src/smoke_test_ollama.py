import json
import urllib.request

url = "http://127.0.0.1:11434/api/generate"
payload = {
    "model": "llama3.2:1b",
    "prompt": "Reply with exactly OK",
    "stream": False,
}

req = urllib.request.Request(
    url,
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)

with urllib.request.urlopen(req) as resp:
    result = json.loads(resp.read().decode("utf-8"))

print(result["response"])
