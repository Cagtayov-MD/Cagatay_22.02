import json
import urllib.request

url = "http://localhost:11434/api/generate"
payload = {
    "model": "qwen2.5vl:7b",   # sende tags'te görünen çalışan model
    "prompt": "say hi",
    "stream": False
}
data = json.dumps(payload).encode("utf-8")

req = urllib.request.Request(
    url,
    data=data,
    headers={"Content-Type": "application/json"},
    method="POST",
)

with urllib.request.urlopen(req, timeout=30) as resp:
    body = resp.read().decode("utf-8", errors="replace")
    print("status:", resp.status)
    print(body[:500])