"""
Based on https://github.com/morioka/tiny-openai-whisper-api
"""

import os
import shutil
from datetime import timedelta
from functools import lru_cache
from typing import Optional

import numpy as np
import uvicorn
import whisper
from fastapi import FastAPI, Form, UploadFile, File
from fastapi import HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def get_whisper_model(whisper_model: str):
    """Get a whisper model from the cache or download it if it doesn't exist"""
    model = whisper.load_model(whisper_model)
    return model


def transcribe(audio_path: str, whisper_model: str, **whisper_args):
    """Transcribe the audio file using whisper"""

    # Get whisper model
    # NOTE: If multiple models are selected, this may keep all of them in memory depending on the cache size
    transcriber = get_whisper_model(whisper_model)

    # Set configs & transcribe
    if whisper_args["temperature_increment_on_fallback"] is not None:
        whisper_args["temperature"] = tuple(
            np.arange(
                whisper_args["temperature"],
                1.0 + 1e-6,
                whisper_args["temperature_increment_on_fallback"],
            )
        )
    else:
        whisper_args["temperature"] = [whisper_args["temperature"]]

    del whisper_args["temperature_increment_on_fallback"]

    transcript = transcriber.transcribe(
        audio_path,
        **whisper_args,
    )

    return transcript


WHISPER_DEFAULT_SETTINGS = {
    "whisper_model": "base",
    "temperature": 0.0,
    "temperature_increment_on_fallback": 0.2,
    "no_speech_threshold": 0.6,
    "logprob_threshold": -1.0,
    "compression_ratio_threshold": 2.4,
    "condition_on_previous_text": True,
    "verbose": False,
    "task": "transcribe",
}

UPLOAD_DIR = "tmp"


@app.post("/v1/audio/transcriptions")
async def transcriptions(
    model: str = Form(...),
    file: UploadFile = File(...),
    response_format: Optional[str] = Form(None),
    temperature: Optional[float] = Form(None),
    settings_override: Optional[dict] = Form(None),
):
    assert model == "whisper-1"

    if file is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Bad Request, bad file"
        )

    if response_format is None:
        response_format = "json"
    if response_format not in ["json", "text", "srt", "verbose_json", "vtt"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Bad Request, bad response_format",
        )

    if temperature is None:
        temperature = 0.0
    if temperature < 0.0 or temperature > 1.0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Bad Request, bad temperature",
        )

    filename = file.filename
    fileobj = file.file
    upload_name = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)

    with open(upload_name, "wb+") as upload_file:
        shutil.copyfileobj(fileobj, upload_file)

    whisper_args = WHISPER_DEFAULT_SETTINGS.copy()
    if settings_override is not None:
        whisper_args.update(settings_override)

    transcript = transcribe(audio_path=upload_name, **whisper_args)

    if response_format in ["text"]:
        return transcript["text"]

    if response_format in ["srt"]:
        ret = ""
        for seg in transcript["segments"]:
            td_s = timedelta(milliseconds=seg["start"] * 1000)
            td_e = timedelta(milliseconds=seg["end"] * 1000)

            t_s = f"{td_s.seconds//3600:02}:{(td_s.seconds//60)%60:02}:{td_s.seconds%60:02}.{td_s.microseconds//1000:03}"
            t_e = f"{td_e.seconds//3600:02}:{(td_e.seconds//60)%60:02}:{td_e.seconds%60:02}.{td_e.microseconds//1000:03}"

            ret += "{}\n{} --> {}\n{}\n\n".format(seg["id"], t_s, t_e, seg["text"])
        ret += "\n"
        return ret

    if response_format in ["vtt"]:
        ret = "WEBVTT\n\n"
        for seg in transcript["segments"]:
            td_s = timedelta(milliseconds=seg["start"] * 1000)
            td_e = timedelta(milliseconds=seg["end"] * 1000)

            t_s = f"{td_s.seconds//3600:02}:{(td_s.seconds//60)%60:02}:{td_s.seconds%60:02}.{td_s.microseconds//1000:03}"
            t_e = f"{td_e.seconds//3600:02}:{(td_e.seconds//60)%60:02}:{td_e.seconds%60:02}.{td_e.microseconds//1000:03}"

            ret += "{} --> {}\n{}\n\n".format(t_s, t_e, seg["text"])
        return ret

    if response_format in ["verbose_json"]:
        transcript.setdefault("task", whisper_args["task"])
        transcript.setdefault("duration", transcript["segments"][-1]["end"])
        return transcript

    return {"text": transcript["text"]}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
