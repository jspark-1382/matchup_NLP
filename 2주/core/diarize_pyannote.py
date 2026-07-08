"""pyannote.audio 기반 화자 분리 선택 실습.

pyannote는 사전 훈련된 화자 분리 파이프라인을 사용합니다.
모델은 models/ 폴더에 로컬로 저장되어 있어 토큰 없이 실행 가능합니다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# 로컬 모델 경로 (models/ 폴더 내)
LOCAL_MODEL_DIR = Path(__file__).parent.parent / "models" / "speaker-diarization-3.1"


def diarize_pyannote(
    audio_path: str | Path,
    hf_token: str | None = None,
    model_name: str | None = None,
) -> pd.DataFrame:
    """pyannote 파이프라인으로 화자 구간을 추출합니다.

    반환 컬럼: start, end, speaker
    """
    # 로컬 모델 경로 결정
    local_path = Path(model_name) if model_name else LOCAL_MODEL_DIR

    try:
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise RuntimeError(
            "pyannote.audio가 설치되어 있지 않습니다. "
            "pip install -r requirements_optional.txt 를 실행해 주세요."
        ) from exc

    # 로컬 모델이 있으면 토큰 없이 로드, 없으면 HF Hub에서 로드
    if local_path.exists():
        pipeline = Pipeline.from_pretrained(str(local_path))
    else:
        # HF Hub에서 로드 (토큰 필요)
        import os
        from dotenv import load_dotenv
        load_dotenv()
        token = hf_token or os.getenv("HF_TOKEN")
        if not token:
            raise RuntimeError(
                "로컬 모델이 없고 HF_TOKEN도 설정되지 않았습니다. "
                "models/ 폴더에 모델을 넣거나 .env 파일에 HF_TOKEN을 설정해 주세요."
            )
        pipeline = Pipeline.from_pretrained(model_name or "pyannote/speaker-diarization-3.1", token=token)

    # torchcodec 미설치 대비: soundfile으로 직접 로드하여 waveform dict 전달
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

    # pyannote 4.x: DiarizeOutput → speaker_diarization 속성 사용
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
