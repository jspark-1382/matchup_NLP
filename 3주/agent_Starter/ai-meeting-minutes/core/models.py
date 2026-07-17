from __future__ import annotations

from typing import Any, TypedDict


class Utterance(TypedDict):
    speaker: str
    start: float
    end: float
    text: str


Transcript = list[Utterance]
GeminiSummary = dict[str, Any]
TraditionalSummary = dict[str, Any]

