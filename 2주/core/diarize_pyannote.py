"""pyannote.audio 기반 화자 분리 선택 실습.

pyannote는 사전 훈련된 화자 분리 파이프라인을 사용합니다.
Hugging Face 토큰이 필요할 수 있으며, 토큰은 코드에 직접 쓰지 말고 환경변수로 전달하세요.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv


DEFAULT_MODEL = "pyannote/speaker-diarization-3.1"


def diarize_pyannote(
    audio_path: str | Path,
    hf_token: str | None = None,
    model_name: str = DEFAULT_MODEL,
) -> pd.DataFrame:
    """pyannote 파이프라인으로 화자 구간을 추출합니다.

    반환 컬럼: start, end, speaker
    """
    load_dotenv()
    token = hf_token or os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN이 없습니다. .env 파일 또는 환경변수에 Hugging Face 토큰을 설정해 주세요."
        )

    try:
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise RuntimeError(
            "pyannote.audio가 설치되어 있지 않습니다. "
            "pip install -r requirements_optional.txt 를 실행해 주세요."
        ) from exc

    pipeline = Pipeline.from_pretrained(model_name, token=token)

    # torchcodec 미설치 대비: soundfile로 직접 로드하여 waveform dict 전달
    import torch
    import soundfile as sf

    audio_data, sample_rate = sf.read(str(audio_path), dtype="float32")
    # (samples,) -> (channels, samples) 변환
    if audio_data.ndim == 1:
        audio_data = audio_data[np.newaxis, :]
    else:
        audio_data = audio_data.T
    waveform = torch.from_numpy(audio_data)
    output = pipeline({"waveform": waveform, "sample_rate": sample_rate})

    # pyannote 4.x: DiiarizeOutput → speaker_diarization 속성 사용
    # pyannote 3.x: Annotation 직접 반환
    if hasattr(output, "speaker_diarization"):
        diarization = output.speaker_diarization
    elif hasattr(output, "annotation"):
        diarization = output.annotation
    else:
        diarization = output

    rows: list[dict[str, object]] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        rows.append(
            {
                "start": float(turn.start),
                "end": float(turn.end),
                "speaker": str(speaker),
            }
        )

    return pd.DataFrame(rows).sort_values("start").reset_index(drop=True)
