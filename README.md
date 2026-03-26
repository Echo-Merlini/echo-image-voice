# echo-image-voice

FastAPI service running on port **7862** that accepts an image, describes it using a vision model, and returns a voice response via Chatterbox TTS.

## How it works

1. Image is sent to LM Studio (local vision model, primary)
2. If LM Studio is unavailable, falls back to Claude API (Anthropic)
3. The description text is sent to Chatterbox TTS to generate a WAV audio response

## Endpoints

- `GET /` — Web UI
- `GET /health` — Health check (LM Studio + Chatterbox status)
- `POST /process` — Upload image + optional prompt → returns WAV audio
- `POST /describe` — Upload image + optional prompt → returns JSON description
- `POST /speak` — Send text → returns WAV audio

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | _(empty)_ | Claude API key for vision fallback when LM Studio is offline |
| `CHATTERBOX_URL` | `http://localhost:5050/speak` | Chatterbox TTS endpoint (hardcoded in server.py — override via env or update server.py) |
| `LM_STUDIO_URL` | `http://localhost:1234/v1/chat/completions` | LM Studio vision model endpoint |

> **Note:** `CHATTERBOX_URL` and `LM_STUDIO_URL` are currently hardcoded in `server.py`. When running in Docker on the NAS, update these values to point to the host machine (e.g. `host.docker.internal:5050`) or pass them as environment variables after adding `os.environ.get()` reads in `server.py`.

## Running with Docker

```bash
docker compose up -d
```

The `docker-compose.yml` includes `extra_hosts: host.docker.internal:host-gateway` so the container can reach services on the host machine.

## Running locally

```bash
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 7862
```
