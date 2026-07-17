from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE = ROOT / "fixtures" / "sample_transcript.json"


def load_demo_transcript(path: str | Path = DEFAULT_FIXTURE) -> list[dict]:
    """강사가 제공한 화자별 대화록 Fixture를 읽는다."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    transcript = data.get("utterances", data)
    required = {"speaker", "start", "end", "text"}
    for index, item in enumerate(transcript):
        missing = required.difference(item)
        if missing:
            raise ValueError(f"대화록 {index}번 항목에 {sorted(missing)}가 없습니다.")
    return transcript


def transcribe_audio(_audio_path: str | Path | None = None, demo_mode: bool = True) -> list[dict]:
    """수업에서는 STT를 새로 만들지 않고 Demo Fixture를 반환한다.

    실제 수업 환경에서는 이 함수 안을 완성된 STT·화자 분리 Adapter로 교체한다.
    """
    if demo_mode:
        return load_demo_transcript()
    raise NotImplementedError("실제 STT Adapter를 연결하거나 Demo Mode를 사용하세요.")

