"""SpeechRecognition 기반 STT 실습 함수."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, TYPE_CHECKING

from .audio_utils import cleanup_temp_files, ensure_wav, export_segment_to_wav

if TYPE_CHECKING:  # pragma: no cover
    import speech_recognition as sr


@dataclass
class TranscriptionResult:
    """STT 결과를 담는 작은 자료형."""

    text: str
    ok: bool
    message: str = ""


def _import_speech_recognition():
    """speech_recognition을 import합니다."""
    try:
        import speech_recognition as sr
    except ImportError as exc:
        raise RuntimeError(
            "SpeechRecognition이 설치되어 있지 않습니다. "
            "pip install -r requirements.txt 를 먼저 실행해 주세요."
        ) from exc
    return sr


def transcribe_file(
    audio_path: str | Path,
    language: str = "ko-KR",
    recognizer: "sr.Recognizer | None" = None,
) -> TranscriptionResult:
    """오디오 파일 하나를 텍스트로 변환합니다.
    Google Web Speech API를 호출하므로 인터넷 연결이 필요합니다.
    긴 회의 녹음은 한 번에 처리하기보다 구간별로 잘라 처리하는 것이 안정적입니다.
    """
    try:
        sr = _import_speech_recognition()
    except RuntimeError as exc:
        return TranscriptionResult(text="", ok=False, message=str(exc))

    recognizer = recognizer or sr.Recognizer()

    try:
        prepared_path = ensure_wav(audio_path)
    except Exception as exc:
        return TranscriptionResult(text="", ok=False, message=f"오디오 준비 실패: {exc}")

    cleanup_prepared = Path(prepared_path) != Path(audio_path)

    try:
        with sr.AudioFile(str(prepared_path)) as source:
            audio = recognizer.record(source)
        text = recognizer.recognize_google(audio, language=language)
        return TranscriptionResult(text=text, ok=True, message="인식 성공")
    except sr.UnknownValueError:
        return TranscriptionResult(text="", ok=False, message="인식 실패: 음성을 이해하지 못했습니다.")
    except sr.RequestError as exc:
        return TranscriptionResult(text="", ok=False, message=f"요청 실패: {exc}")
    except FileNotFoundError:
        return TranscriptionResult(text="", ok=False, message=f"파일을 찾을 수 없습니다: {audio_path}")
    except Exception as exc:  # pragma: no cover - 오디오/네트워크 환경에 따라 달라짐
        return TranscriptionResult(text="", ok=False, message=f"처리 실패: {type(exc).__name__}: {exc}")
    finally:
        if cleanup_prepared:
            cleanup_temp_files([prepared_path])


def transcribe_microphone(
    language: str = "ko-KR",
    phrase_time_limit: int | None = 8,
    ambient_noise_seconds: float = 0.8,
) -> TranscriptionResult:
    """마이크 입력을 받아 텍스트로 변환합니다.

    운영체제의 마이크 권한과 PyAudio 설치가 필요할 수 있습니다.
    """
    try:
        sr = _import_speech_recognition()
    except RuntimeError as exc:
        return TranscriptionResult(text="", ok=False, message=str(exc))

    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=ambient_noise_seconds)
            audio = recognizer.listen(source, phrase_time_limit=phrase_time_limit)
        text = recognizer.recognize_google(audio, language=language)
        return TranscriptionResult(text=text, ok=True, message="인식 성공")
    except sr.UnknownValueError:
        return TranscriptionResult(text="", ok=False, message="인식 실패: 음성을 이해하지 못했습니다.")
    except sr.RequestError as exc:
        return TranscriptionResult(text="", ok=False, message=f"요청 실패: {exc}")
    except Exception as exc:  # pragma: no cover - 마이크 환경에 따라 달라짐
        return TranscriptionResult(text="", ok=False, message=f"마이크 처리 실패: {type(exc).__name__}: {exc}")


def load_demo_transcript() -> str:
    """오프라인 데모용 회의 문장을 불러옵니다."""
    path = Path(__file__).resolve().parents[1] / "data" / "sample_transcript.txt"
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return "안녕하세요. 오늘 회의는 프로젝트 진행 상황을 공유하고 다음 일정을 정하는 자리입니다."


def _iter_segment_rows(segments: Any) -> Iterable[dict[str, Any]]:
    """DataFrame 또는 dict 리스트를 공통 반복 형태로 바꿉니다."""
    if hasattr(segments, "to_dict"):
        yield from segments.to_dict("records")
    else:
        yield from segments


def transcribe_segments(
    audio_path: str | Path,
    segments: Any,
    language: str = "ko-KR",
    start_col: str = "start",
    end_col: str = "end",
    speaker_col: str = "speaker",
    min_duration: float = 0.4,
) -> list[dict[str, Any]]:
    """화자 구간별로 오디오를 잘라 STT를 수행합니다.

    반환값은 `start`, `end`, `speaker`, `text`, `ok`, `message`를 가진 dict 리스트입니다.
    """
    rows: list[dict[str, Any]] = []
    temp_files: list[Path] = []

    try:
        for row in _iter_segment_rows(segments):
            start = float(row[start_col])
            end = float(row[end_col])
            speaker = str(row.get(speaker_col, "SPEAKER_00"))

            if end - start < min_duration:
                rows.append(
                    {
                        "start": start,
                        "end": end,
                        "speaker": speaker,
                        "text": "",
                        "ok": False,
                        "message": "너무 짧은 구간이라 건너뜀",
                    }
                )
                continue

            segment_path = export_segment_to_wav(audio_path, start, end)
            temp_files.append(segment_path)
            result = transcribe_file(segment_path, language=language)
            rows.append(
                {
                    "start": start,
                    "end": end,
                    "speaker": speaker,
                    "text": result.text,
                    "ok": result.ok,
                    "message": result.message,
                }
            )
    finally:
        cleanup_temp_files(temp_files)

    return rows
