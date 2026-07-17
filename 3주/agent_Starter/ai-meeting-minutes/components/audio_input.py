import streamlit as st


def render_audio_input():
    """TODO: Web Developer Agent가 업로드·녹음·재생 UI를 완성하세요."""
    meeting_title = st.text_input("회의 제목")
    uploaded = st.file_uploader("음성 파일 업로드", type=["wav", "mp3", "m4a"])
    if uploaded:
        st.audio(uploaded)
    return meeting_title, uploaded

