"""실습 3: 최소 Streamlit 앱.

사용법:
    streamlit run exercises/03_streamlit_minimal.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from core.audio_utils import cleanup_temp_files, save_uploaded_file  # noqa: E402
from core.stt import load_demo_transcript, transcribe_file  # noqa: E402


st.title("AI 회의록 - 최소 앱")

uploaded = st.file_uploader("오디오 업로드", type=["wav", "flac", "aiff", "aif", "mp3", "m4a"])
use_demo_text = st.checkbox("오프라인 데모 텍스트 사용", value=False)

if uploaded:
    st.audio(uploaded)

if st.button("STT 실행"):
    if not uploaded:
        st.warning("먼저 오디오 파일을 업로드하세요.")
        st.stop()

    if use_demo_text:
        text = load_demo_transcript()
        st.text_area("인식 결과", text, height=180)
        st.stop()

    temp_files = []
    try:
        audio_path = save_uploaded_file(uploaded)
        temp_files.append(audio_path)
        result = transcribe_file(audio_path, language="ko-KR")
        if result.ok:
            st.success(result.message)
        else:
            st.warning(result.message)
        st.text_area("인식 결과", result.text, height=180)
    finally:
        cleanup_temp_files(temp_files)
