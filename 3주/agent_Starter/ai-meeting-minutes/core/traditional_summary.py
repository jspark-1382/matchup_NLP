from __future__ import annotations


def summarize_traditional(
    transcript: list[dict],
    *,
    sentence_count: int = 3,
    keyword_count: int = 5,
) -> dict:
    """TODO: Traditional Summary Agent가 구현할 함수입니다.

    TF-IDF → 코사인 유사도 → TextRank → 상위 문장 선택 →
    원래 시간 순서 재배치 흐름으로 구현하세요.
    """
    return {
        "method": "textrank",
        "summary_sentences": [],
        "keywords": [],
    }

