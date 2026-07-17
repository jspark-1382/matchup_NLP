from __future__ import annotations

import streamlit as st

from components.audio_input import render_audio_input
from components.summary_view import render_gemini_summary, render_traditional_summary
from components.transcript_view import render_transcript
from core.gemini_summary import generate_gemini_summary
from core.traditional_summary import summarize_traditional
from core.transcribe import load_demo_transcript


st.set_page_config(page_title="AI 회의록 Starter", page_icon="📝", layout="wide")
st.title("📝 AI 회의록 Starter")
st.warning("TODO: Web Developer Agent가 이 화면을 완성합니다.")

meeting_title, audio = render_audio_input()
if st.button("Demo 대화록 불러오기"):
    transcript = load_demo_transcript()
    st.session_state["transcript"] = transcript

transcript = st.session_state.get("transcript", [])
if transcript:
    transcript_tab, gemini_tab, traditional_tab = st.tabs(
        ["화자별 대화록", "Gemini 요약", "전통적 요약"]
    )
    with transcript_tab:
        render_transcript(transcript)
    with gemini_tab:
        render_gemini_summary(generate_gemini_summary(transcript, use_mock=True))
    with traditional_tab:
        render_traditional_summary(summarize_traditional(transcript))

