import streamlit as st


def render_gemini_summary(data: dict) -> None:
    """TODO: 요약·결정사항·액션 아이템·미해결 질문을 표시하세요."""
    st.json(data)


def render_traditional_summary(data: dict) -> None:
    """TODO: 중요 문장·키워드·점수를 표시하세요."""
    st.json(data)


def render_comparison() -> None:
    """TODO: 두 방식의 장점과 한계를 나란히 설명하세요."""
    st.info("비교 설명을 구현하세요.")

