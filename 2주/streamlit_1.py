import streamlit as st
from pathlib import Path
from core.stt import transcribe_file

st.title("AI 회의록")
file = st.file_uploader("오디오 파일 업로드", type=["wav", "flac", "aiff", "mp3"])


if st.button("STT 실행"):
    audio_dir = Path("uploaded_audio") / file.name
    audio_dir.mkdir(exist_ok=True)
    audio_path = audio_dir / file.name
    audio_path.write_bytes(file.getvalue()) # 파일을 경로에 저장

    text = transcribe_file(audio_path, language="ko-KR") # 오디오 Path를 통해 파일 불러오기
    st.text_area("인식 결과", value=text, height=200)