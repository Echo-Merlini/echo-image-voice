#!/usr/bin/env python3
"""
Echo Image+Voice Service — port 7862
Accepts image + text, describes image via LM Studio vision (fallback: Claude API), speaks via Chatterbox TTS.
"""

import io
import os
import base64
import httpx
import logging
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import Response, HTMLResponse

# Load .env if present
_env_file = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(_env_file):
    for line in open(_env_file):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("image-voice")

LM_STUDIO_URL = os.environ.get("LM_STUDIO_URL", "http://localhost:1234/v1/chat/completions")
CHATTERBOX_URL = os.environ.get("CHATTERBOX_URL", "http://localhost:5050/speak")
VISION_MODEL = "google/gemma-3-4b"
CLAUDE_MODEL = "claude-haiku-4-5-20251001"
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

app = FastAPI(title="Echo Image+Voice Service")


async def describe_image_lmstudio(image_bytes: bytes, prompt: str) -> str:
    b64 = base64.b64encode(image_bytes).decode()
    payload = {
        "model": VISION_MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            {"type": "text", "text": prompt or "Describe this image concisely in 2-3 sentences."}
        ]}],
        "max_tokens": 300,
        "temperature": 0.7,
    }
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(LM_STUDIO_URL, json=payload)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()


async def describe_image_claude(image_bytes: bytes, prompt: str) -> str:
    if not ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    b64 = base64.b64encode(image_bytes).decode()
    msg = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
            {"type": "text", "text": prompt or "Describe this image concisely in 2-3 sentences."}
        ]}]
    )
    return msg.content[0].text.strip()


async def describe_image(image_bytes: bytes, prompt: str = "") -> str:
    try:
        result = await describe_image_lmstudio(image_bytes, prompt)
        log.info("Vision: LM Studio")
        return result
    except Exception as e:
        log.warning(f"LM Studio unavailable ({e}), falling back to Claude API")
        result = await describe_image_claude(image_bytes, prompt)
        log.info("Vision: Claude API (fallback)")
        return result


async def synthesize_voice(text: str) -> bytes:
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(CHATTERBOX_URL, json={"text": text})
        r.raise_for_status()
        return r.content


@app.get("/health")
async def health():
    lm_ok = False
    chatterbox_ok = False
    try:
        async with httpx.AsyncClient(timeout=3) as c:
            await c.get("http://localhost:1234/v1/models")
            lm_ok = True
    except Exception:
        pass
    try:
        async with httpx.AsyncClient(timeout=3) as c:
            await c.get(CHATTERBOX_URL.replace("/speak", "/health"))
            chatterbox_ok = True
    except Exception:
        pass
    return {
        "status": "ok",
        "vision_model": VISION_MODEL,
        "lm_studio": lm_ok,
        "chatterbox": chatterbox_ok,
        "claude_fallback": bool(ANTHROPIC_API_KEY),
    }


@app.post("/process")
async def process(
    image: UploadFile = File(...),
    prompt: str = Form(default=""),
):
    image_bytes = await image.read()
    log.info(f"Processing image: {image.filename}, prompt: {prompt[:60]}")

    try:
        description = await describe_image(image_bytes, prompt)
        log.info(f"Description: {description[:80]}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Vision model error: {e}")

    try:
        audio = await synthesize_voice(description)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Voice synthesis error: {e}")

    return Response(
        content=audio,
        media_type="audio/wav",
        headers={"X-Description": description[:200]}
    )


@app.post("/describe")
async def describe_only(
    image: UploadFile = File(...),
    prompt: str = Form(default=""),
):
    image_bytes = await image.read()
    description = await describe_image(image_bytes, prompt)
    return {"description": description}


@app.post("/speak")
async def speak_only(text: str = Form(...)):
    audio = await synthesize_voice(text)
    return Response(content=audio, media_type="audio/wav")




@app.get("/", response_class=HTMLResponse)
async def ui():
    return HTML


HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Echo — Image + Voice</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0f0f11;color:#e1e1e6;min-height:100vh;padding:2rem;max-width:680px;margin:0 auto}
  h1{font-size:1.5rem;font-weight:600;margin-bottom:.25rem}
  .subtitle{color:#888;font-size:.875rem;margin-bottom:2rem}
  .card{background:#1a1a1f;border:1px solid #2a2a35;border-radius:14px;padding:1.5rem;margin-bottom:1.5rem}
  label{font-size:.75rem;color:#888;display:block;margin-bottom:.4rem}
  .drop-zone{border:2px dashed #2a2a35;border-radius:10px;padding:2rem;text-align:center;cursor:pointer;transition:all .2s;margin-bottom:1rem}
  .drop-zone.dragover,.drop-zone:hover{border-color:#4f46e5;background:#1e1e2e}
  .drop-zone img{max-width:100%;max-height:200px;border-radius:8px;display:none;margin:0 auto}
  .drop-zone p{color:#555;font-size:.85rem}
  input[type=text]{width:100%;background:#0f0f11;border:1px solid #2a2a35;color:#e1e1e6;border-radius:8px;padding:.55rem .85rem;font-size:.9rem;outline:none;margin-bottom:1rem}
  input[type=text]:focus{border-color:#4f46e5}
  button{background:#4f46e5;color:#fff;border:none;border-radius:8px;padding:.6rem 1.4rem;font-size:.9rem;font-weight:500;cursor:pointer;transition:background .15s;width:100%}
  button:hover{background:#6366f1}
  button:disabled{background:#2a2a35;color:#555;cursor:not-allowed}
  .result{margin-top:1.25rem;display:none}
  .description{background:#0f0f11;border:1px solid #2a2a35;border-radius:8px;padding:.85rem;font-size:.85rem;color:#c4c4d0;margin-bottom:.75rem;line-height:1.5}
  audio{width:100%;border-radius:8px}
  .status{font-size:.78rem;color:#6366f1;margin-top:.6rem;min-height:1rem}
  .section-title{font-size:.85rem;font-weight:600;color:#a5b4fc;margin-bottom:1rem}
  .speak-row{display:flex;gap:.5rem}
  .speak-row input{margin-bottom:0;flex:1}
  .speak-row button{width:auto;flex-shrink:0}
</style>
</head>
<body>
<h1>🎙️ Echo — Image + Voice</h1>
<p class="subtitle">Upload an image to get a voice description using the active voice clone</p>

<div class="card">
  <div class="section-title">🖼️ Image → Voice</div>
  <label>Image (drag & drop or click)</label>
  <div class="drop-zone" id="dropZone" onclick="document.getElementById('fileInput').click()">
    <img id="preview" alt="preview">
    <p id="dropText">Drop image here or click to upload</p>
  </div>
  <input type="file" id="fileInput" accept="image/*" style="display:none" onchange="previewFile(this)">
  <label>Optional prompt (leave blank for auto-description)</label>
  <input type="text" id="prompt" placeholder="e.g. What's happening in this image?">
  <button id="processBtn" onclick="process()" disabled>🎙️ Describe & Speak</button>
  <div class="status" id="status"></div>
  <div class="result" id="result">
    <div class="description" id="description"></div>
    <audio id="audio" controls></audio>
  </div>
</div>

<div class="card">
  <div class="section-title">💬 Text → Voice</div>
  <label>Text to speak</label>
  <div class="speak-row">
    <input type="text" id="speakText" placeholder="Type something…" onkeydown="if(event.key==='Enter')speakText()">
    <button onclick="speakText()" style="width:auto">Speak</button>
  </div>
  <div class="status" id="speakStatus"></div>
  <audio id="speakAudio" controls style="margin-top:.75rem;width:100%;display:none"></audio>
</div>

<script>
let selectedFile = null;

function previewFile(input) {
  if (!input.files[0]) return;
  selectedFile = input.files[0];
  const reader = new FileReader();
  reader.onload = e => {
    const img = document.getElementById('preview');
    img.src = e.target.result;
    img.style.display = 'block';
    document.getElementById('dropText').style.display = 'none';
    document.getElementById('processBtn').disabled = false;
  };
  reader.readAsDataURL(input.files[0]);
}

const dz = document.getElementById('dropZone');
dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('dragover'); });
dz.addEventListener('dragleave', () => dz.classList.remove('dragover'));
dz.addEventListener('drop', e => {
  e.preventDefault(); dz.classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith('image/')) {
    document.getElementById('fileInput').files = e.dataTransfer.files;
    previewFile(document.getElementById('fileInput'));
  }
});

async function process() {
  if (!selectedFile) return;
  const btn = document.getElementById('processBtn');
  const status = document.getElementById('status');
  btn.disabled = true;
  status.textContent = 'Analyzing image…';
  document.getElementById('result').style.display = 'none';

  const fd = new FormData();
  fd.append('image', selectedFile);
  fd.append('prompt', document.getElementById('prompt').value);

  try {
    const res = await fetch('/process', { method: 'POST', body: fd });
    if (!res.ok) { const e = await res.json(); throw new Error(e.detail); }
    const description = res.headers.get('X-Description') || '';
    const blob = await res.blob();
    document.getElementById('description').textContent = description;
    document.getElementById('audio').src = URL.createObjectURL(blob);
    document.getElementById('result').style.display = 'block';
    status.textContent = '';
  } catch(e) {
    status.textContent = '❌ ' + e.message;
  }
  btn.disabled = false;
}

async function speakText() {
  const text = document.getElementById('speakText').value.trim();
  if (!text) return;
  const status = document.getElementById('speakStatus');
  status.textContent = 'Generating voice…';
  const fd = new FormData();
  fd.append('text', text);
  try {
    const res = await fetch('/speak', { method: 'POST', body: fd });
    if (!res.ok) throw new Error('Voice error');
    const blob = await res.blob();
    const audio = document.getElementById('speakAudio');
    audio.src = URL.createObjectURL(blob);
    audio.style.display = 'block';
    audio.play();
    status.textContent = '';
  } catch(e) {
    status.textContent = '❌ ' + e.message;
  }
}
</script>
</body>
</html>"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7862, log_level="info")
