from __future__ import annotations

from typing import Any


REQUIRED_KEYS = {"summary", "key_points", "decisions", "action_items", "open_questions"}


def validate_summary(data: dict[str, Any]) -> dict[str, Any]:
    """TODO: Gemini 응답의 필수 키와 내부 항목을 검증하세요."""
    return data


def generate_gemini_summary(
    transcript: list[dict],
    *,
    use_mock: bool = True,
    api_key: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """TODO: Gemini Summary Agent가 구현할 함수입니다.

    Demo Mode에서는 fixtures/sample_gemini_summary.json을 읽고,
    API Mode에서는 환경변수의 키로 구조화된 JSON 요약을 생성하세요.
    """
    return {
        "summary": "아직 구현 전입니다.",
        "key_points": [],
        "decisions": [],
        "action_items": [],
        "open_questions": [],
    }

