"""화자 분리 결과와 STT 결과를 연결하는 파이프라인."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

from .diarize import diarize_kmeans
from .diarize_pyannote import diarize_pyannote
from .stt import load_demo_transcript, transcribe_segments


DiarizationMethod = Literal["kmeans", "pyannote"]


def diarize_audio(
    audio_path: str | Path,
    method: DiarizationMethod = "kmeans",
    n_speakers: int = 2,
    chunk_seconds: float = 5.0,
) -> pd.DataFrame:
    """선택한 방식으로 화자 분리 시간표를 얻습니다."""
    if method == "kmeans":
        return diarize_kmeans(
            audio_path=audio_path,
            n_speakers=n_speakers,
            chunk_seconds=chunk_seconds,
            merge=True,
        )
    if method == "pyannote":
        return diarize_pyannote(audio_path)
    raise ValueError(f"지원하지 않는 화자 분리 방식입니다: {method}")


def build_diarized_transcript(
    audio_path: str | Path,
    method: DiarizationMethod = "kmeans",
    language: str = "ko-KR",
    n_speakers: int = 2,
    chunk_seconds: float = 5.0,
    use_demo_text: bool = False,
) -> pd.DataFrame:
    """화자 구간을 얻고, 각 구간별 STT 결과를 붙입니다."""
    diarization = diarize_audio(
        audio_path=audio_path,
        method=method,
        n_speakers=n_speakers,
        chunk_seconds=chunk_seconds,
    )

    if use_demo_text:
        demo_sentences = [line.strip() for line in load_demo_transcript().splitlines() if line.strip()]
        if not demo_sentences:
            demo_sentences = [load_demo_transcript()]
        rows = []
        for idx, row in diarization.iterrows():
            rows.append(
                {
                    "start": row["start"],
                    "end": row["end"],
                    "speaker": row["speaker"],
                    "text": demo_sentences[idx % len(demo_sentences)],
                    "ok": True,
                    "message": "오프라인 데모 텍스트",
                }
            )
        return pd.DataFrame(rows)

    rows = transcribe_segments(audio_path, diarization, language=language)
    return pd.DataFrame(rows)


def format_minutes(df: pd.DataFrame) -> str:
    """회의록 표를 텍스트 회의록 형태로 바꿉니다."""
    if df.empty:
        return ""

    lines: list[str] = []
    for _, row in df.iterrows():
        start = float(row["start"])
        end = float(row["end"])
        speaker = row.get("speaker", "SPEAKER_00")
        text = str(row.get("text", "")).strip()
        if not text:
            text = f"({row.get('message', '텍스트 없음')})"
        lines.append(f"[{start:06.2f}-{end:06.2f}] {speaker}: {text}")
    return "\n".join(lines)
