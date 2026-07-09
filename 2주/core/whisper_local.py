"""로컬 Whisper 모델 기반 STT 실습.

faster-whisper를 사용하며, 모델은 models/whisper/ 폴더에
로컬로 저장되어 있어 토큰 없이 실행 가능합니다.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


# 로컬 모델 경로 (models/whisper/ 폴더 내)
LOCAL_MODEL_DIR = Path(__file__).parent.parent / "models" / "whisper"

# 사용 가능한 모델 크기
AVAILABLE_MODELS = ["tiny", "base", "small", "medium", "large-v3"]


def transcribe_whisper(
    audio_path: str | Path,
    model_size: str = "base",
    language: str | None = "ko",
) -> pd.DataFrame:
    """로컬 Whisper 모델로 오디오를 텍스트로 변환합니다.

    반환 컬럼: start, end, text
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "faster-whisper가 설치되어 있지 않습니다. "
            "pip install -r requirements_optional.txt 를 실행해 주세요."
        ) from exc

    # 로컬 모델 경로 결정
    local_model_path = LOCAL_MODEL_DIR / model_size

    if local_model_path.exists():
        model = WhisperModel(str(local_model_path), device="cpu", compute_type="int8")
    else:
        # HF Hub에서 로드 (토큰 필요할 수 있음)
        model = WhisperModel(model_size, device="cpu", compute_type="int8")

    segments, _ = model.transcribe(
        str(audio_path),
        language=language,
        beam_size=5,
    )

    rows: list[dict[str, object]] = []
    for segment in segments:
        rows.append(
            {
                "start": float(segment.start),
                "end": float(segment.end),
                "text": segment.text.strip(),
            }
        )

    return pd.DataFrame(rows)
