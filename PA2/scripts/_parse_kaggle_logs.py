import json
import os

from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()
raw = api.kernels_logs("salmanrifa/pa2-sft-dpo-pipeline") or ""
text = raw
try:
    events = json.loads(raw)
    parts = []
    for e in events:
        if isinstance(e, dict) and "data" in e:
            parts.append(e["data"])
        else:
            parts.append(str(e))
    text = "".join(parts)
except Exception:
    pass

out = "outputs/kaggle_logs_parsed.txt"
os.makedirs("outputs", exist_ok=True)
with open(out, "w", encoding="utf-8") as f:
    f.write(text)
print("wrote", out, "chars", len(text))
lines = text.splitlines()
keys = (
    "Error",
    "error",
    "Traceback",
    "Exception",
    "OOM",
    "CUDA",
    "Wrote",
    "Generated",
    "step=",
    "Loading",
    "Done",
    "FAILED",
    "RuntimeError",
    "FileNotFound",
    "No such",
    "exit",
)
for i, line in enumerate(lines):
    if any(k in line for k in keys):
        print(f"{i}: {line[:400]}")
print("---- LAST 80 ----")
for line in lines[-80:]:
    print(line[:400])
