"""Regression tests for the optional `language` form field.

Proves:
- omitted language keeps the existing auto-detect behavior,
- language=pl reaches the faster-whisper transcribe call,
- an invalid language is rejected deterministically (400),
- existing validation/response behavior is unchanged.
"""

import asyncio
import io

from fastapi import UploadFile
from fastapi.testclient import TestClient

import main


class FakeSegment:
    def __init__(self):
        self.id = 0
        self.start = 0.0
        self.end = 0.5
        self.text = " test"


class FakeInfo:
    language = None
    language_probability = None
    duration = 0.5


def make_client(monkeypatch, tmp_path, calls):
    """TestClient with main.transcribe replaced by a recording stub."""

    def fake_transcribe(**kwargs):
        calls.append(kwargs)
        return [FakeSegment()], FakeInfo()

    monkeypatch.setattr(main, "transcribe", fake_transcribe)
    monkeypatch.chdir(tmp_path)
    return TestClient(main.app)


def post_transcription(client, data=None):
    payload = {"model": "whisper-1"}
    if data:
        payload.update(data)
    return client.post(
        "/v1/audio/transcriptions",
        files={"file": ("audio.wav", b"fake-wav-bytes", "audio/wav")},
        data=payload,
    )


def test_no_language_keeps_auto_detect(monkeypatch, tmp_path):
    calls = []
    client = make_client(monkeypatch, tmp_path, calls)

    response = post_transcription(client)

    assert response.status_code == 200
    assert response.json() == {"text": " test"}
    assert len(calls) == 1
    assert "language" not in calls[0]


def test_language_pl_reaches_transcribe_call(monkeypatch, tmp_path):
    calls = []
    client = make_client(monkeypatch, tmp_path, calls)

    response = post_transcription(client, {"language": "pl"})

    assert response.status_code == 200
    assert len(calls) == 1
    assert calls[0]["language"] == "pl"


def test_invalid_language_returns_400_and_skips_transcribe(monkeypatch, tmp_path):
    calls = []
    client = make_client(monkeypatch, tmp_path, calls)

    response = post_transcription(client, {"language": "xx-totally-invalid"})

    assert response.status_code == 400
    assert "language" in response.json()["detail"]
    assert calls == []


def test_empty_language_string_behaves_as_omitted(monkeypatch, tmp_path):
    """The multipart stack delivers an empty form value as None, so
    `language=` is equivalent to omitted -> auto-detect stays intact."""
    calls = []
    client = make_client(monkeypatch, tmp_path, calls)

    response = post_transcription(client, {"language": ""})

    assert response.status_code == 200
    assert len(calls) == 1
    assert "language" not in calls[0]


def test_uppercase_language_is_rejected_like_faster_whisper_would(
    monkeypatch, tmp_path
):
    """faster-whisper Tokenizer validates case-sensitively; mirror that."""
    calls = []
    client = make_client(monkeypatch, tmp_path, calls)

    response = post_transcription(client, {"language": "PL"})

    assert response.status_code == 400
    assert calls == []


def test_openapi_schema_declares_optional_language():
    schema = main.app.openapi()
    body_schema_name = next(
        name
        for name in schema["components"]["schemas"]
        if name.startswith("Body_transcriptions")
    )
    properties = schema["components"]["schemas"][body_schema_name]["properties"]
    assert "language" in properties
    assert properties["language"]["title"] == "Language"
    assert {"type": "string"} in properties["language"]["anyOf"]
    assert {"type": "null"} in properties["language"]["anyOf"]


def test_default_json_response_format_unchanged(monkeypatch, tmp_path):
    calls = []
    client = make_client(monkeypatch, tmp_path, calls)

    response = post_transcription(client)

    assert response.status_code == 200
    assert set(response.json().keys()) == {"text"}


def test_explicit_text_response_format_unchanged(monkeypatch, tmp_path):
    """Existing upstream quirk preserved: response_format=text still returns
    the JSON dict shape (documented here as regression protection)."""
    calls = []
    client = make_client(monkeypatch, tmp_path, calls)

    response = post_transcription(client, {"response_format": "text"})

    assert response.status_code == 200
    assert response.json() == {"text": " test"}


def test_bad_response_format_still_400(monkeypatch, tmp_path):
    calls = []
    client = make_client(monkeypatch, tmp_path, calls)

    response = post_transcription(client, {"response_format": "bogus"})

    assert response.status_code == 400
    assert response.json()["detail"] == "Bad Request, bad response_format"
    assert calls == []


def test_language_field_takes_precedence_over_settings_override(monkeypatch, tmp_path):
    """The dedicated API field must win over the generic settings_override
    dict, even when both carry a 'language' key. Called directly because
    settings_override cannot be populated over multipart HTTP anyway."""
    calls = []
    make_client(monkeypatch, tmp_path, calls)

    upload = UploadFile(file=io.BytesIO(b"fake-wav-bytes"), filename="audio.wav")

    async def call_route():
        return await main.transcriptions(
            model="whisper-1",
            file=upload,
            response_format=None,
            temperature=None,
            language="pl",
            settings_override={"language": "en"},
        )

    asyncio.run(call_route())

    assert calls[0]["language"] == "pl"


def test_bad_temperature_still_400(monkeypatch, tmp_path):
    calls = []
    client = make_client(monkeypatch, tmp_path, calls)

    response = post_transcription(client, {"temperature": 2.5})

    assert response.status_code == 400
    assert response.json()["detail"] == "Bad Request, bad temperature"
    assert calls == []
