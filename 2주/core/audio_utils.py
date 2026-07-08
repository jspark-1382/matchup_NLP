"""오디오 파일 저장, 변환, 구간 자르기 유틸리티."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import BinaryIO, Iterable


SUPPORTED_BY_SPEECH_RECOGNITION = {".wav", ".aiff", ".aif", ".flac"}


def format_seconds(seconds: float) -> str:
    """초 단위 시간을 `MM:SS.xx` 형태로 바꿉니다."""
    seconds = max(0.0, float(seconds))
    minutes = int(seconds // 60)
    remain = seconds - minutes * 60
    return f"{minutes:02d}:{remain:05.2f}"


def save_binary_to_temp(data: bytes, suffix: str = ".wav") -> Path:
    """bytes 데이터를 임시 파일로 저장하고 경로를 반환합니다."""
    if suffix and not suffix.startswith("."):
        suffix = f".{suffix}"
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    with handle:
        handle.write(data)
    return Path(handle.name)


def save_uploaded_file(uploaded_file: BinaryIO, suffix: str | None = None) -> Path:
    """Streamlit 업로드 객체 또는 파일 객체를 임시 파일로 저장합니다."""
    name = getattr(uploaded_file, "name", "uploaded.wav")
    guessed_suffix = Path(name).suffix or ".wav"
    target_suffix = suffix or guessed_suffix

    if hasattr(uploaded_file, "getvalue"):
        data = uploaded_file.getvalue()
    else:
        data = uploaded_file.read()
    return save_binary_to_temp(data, target_suffix)


def ensure_wav(audio_path: str | os.PathLike[str]) -> Path:
    """STT에 넣기 좋은 wav 파일을 준비합니다.

    wav/aiff/flac 파일이면 원본 경로를 그대로 반환합니다.
    mp3/m4a 등은 pydub과 ffmpeg를 이용해 임시 wav로 변환합니다.
    """
    path = Path(audio_path)
    if path.suffix.lower() in SUPPORTED_BY_SPEECH_RECOGNITION:
        return path

    try:
        from pydub import AudioSegment
    except ImportError as exc:
        raise RuntimeError(
            "mp3/m4a 변환에는 pydub와 ffmpeg가 필요합니다. "
            "wav 파일을 사용하거나 requirements.txt를 설치해 주세요."
        ) from exc

    output = Path(tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name)
    try:
        audio = AudioSegment.from_file(path)
        audio.export(output, format="wav")
    except Exception as exc:  # pragma: no cover - ffmpeg 환경에 따라 달라짐
        raise RuntimeError(
            "오디오 변환에 실패했습니다. mp3/m4a 파일이면 ffmpeg 설치 여부를 확인해 주세요."
        ) from exc
    return output


def export_segment_to_wav(
    audio_path: str | os.PathLike[str],
    start: float,
    end: float,
    output_path: str | os.PathLike[str] | None = None,
) -> Path:
    """오디오의 일부 구간을 잘라 wav 파일로 저장합니다."""
    try:
        from pydub import AudioSegment
    except ImportError as exc:
        raise RuntimeError("구간 자르기에는 pydub가 필요합니다.") from exc

    if end <= start:
        raise ValueError("end는 start보다 커야 합니다.")

    source = AudioSegment.from_file(audio_path)
    start_ms = int(start * 1000)
    end_ms = int(end * 1000)

    if output_path is None:
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name

    output = Path(output_path)
    source[start_ms:end_ms].export(output, format="wav")
    return output


def cleanup_temp_files(paths: Iterable[str | os.PathLike[str]]) -> None:
    """임시 파일들을 조용히 삭제합니다."""
    for path in paths:
        try:
            Path(path).unlink(missing_ok=True)
        except Exception:
            pass
