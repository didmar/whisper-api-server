# Whisper API server

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Drop-in replacement for the OpenAI's [Whisper API](https://platform.openai.com/docs/models/whisper) using the same API but running locally.

Modified from https://github.com/morioka/tiny-openai-whisper-api.

## Setup

### With Docker compose (recommended)

Install [Docker](https://docs.docker.com/engine/install/), then run:

```bash
docker build -t didmar/whisper-api-server .
docker run -d -p 127.0.0.1:8000:8000 -v .cache:/root/.cache:rw didmar/whisper-api-server --workers 1
# Customize number of workers is needs to handle multiple requests in parallel
```

Or use Docker compose:

```bash
docker compose up --build -d
```

### Without Docker

Requires python >= 3.10, pip and virtualenv.

```bash
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
python3 main.py
```

## Usage

Check out the API documentation at http://localhost:8000/docs.

For example:
```bash
curl -sSfLO https://upload.wikimedia.org/wikipedia/commons/2/2b/American_English_sound_%22h%22_%28female%29.wav

curl --request POST \
  --url "http://localhost:8000/v1/audio/transcriptions" \
  --header 'Content-Type: multipart/form-data' \
  --form file=@American_English_sound_%22h%22_%28female%29.wav \
  --form model=whisper-1
# {"text":" Hot hole high. Hu. Hu. Hu. Behind. Hu. Hu."}
```

### Language

An optional `language` form field accepts an ISO 639-1 language code (for
example `pl`) and forces faster-whisper to transcribe with that language
instead of running automatic language detection.

- Omitting the field keeps automatic detection.
- An invalid code is rejected deterministically with `400 Bad Request`.

```bash
curl --request POST \
  --url "http://localhost:8000/v1/audio/transcriptions" \
  --header 'Content-Type: multipart/form-data' \
  --form file=@polski.wav \
  --form model=whisper-1 \
  --form language=pl
```

## Development setup

We use [uv](https://docs.astral.sh/uv/) to manage dependencies,
[Ruff](https://docs.astral.sh/ruff/) for linting and formatting, and
[pytest](https://docs.pytest.org/) for tests.

```bash
# Install dependencies (including development dependencies)
uv sync --dev

# Run tests
uv run pytest

# Setup pre-commit hooks (See https://pre-commit.com/)
pre-commit install
```
