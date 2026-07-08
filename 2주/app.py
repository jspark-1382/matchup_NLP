"""2주차 통합 실습 앱: 오디오 업로드 → STT → 화자 분리 → 회의록 표."""

from __future__ import annotations

import traceback

import streamlit as st

from core.audio_utils import cleanup_temp_files, save_uploaded_file
from core.diarize import diarize_kmeans, format_timeline
from core.pipeline import build_diarized_transcript, format_minutes
from core.stt import load_demo_transcript, transcribe_file


st.set_page_config(page_title="AI 회의록 실습", page_icon="🎙️", layout="wide")

st.title("🎙️ AI 회의록 실습")
st.caption("오디오를 업로드하고, STT와 화자 분리를 연결해 회의록의 뼈대를 만듭니다.")

with st.sidebar:
    st.header("실습 설정")
    language = st.selectbox("STT 언어", ["ko-KR", "en-US", "ja-JP"], index=0)
    use_demo_text = st.checkbox(
        "오프라인 데모 텍스트 사용",
        value=False,
        help="네트워크 없이 화면 흐름만 확인하고 싶을 때 사용합니다.",
    )
    st.divider()
    st.subheader("KMeans 화자 분리")
    n_speakers = st.number_input("예상 화자 수", min_value=1, max_value=6, value=2, step=1)
    chunk_seconds = st.slider("오디오 자르기 간격(초)", min_value=2.0, max_value=15.0, value=5.0, step=1.0)

uploaded = st.file_uploader(
    "오디오 파일 업로드",
    type=["wav", "flac", "aiff", "aif", "mp3", "m4a"],
    help="wav가 가장 안정적입니다. mp3/m4a는 pydub와 ffmpeg가 필요할 수 있습니다.",
)

if uploaded is None:
    st.info("먼저 회의 녹음 파일을 업로드하세요. `data/sample.wav` 또는 `data/meeting_synthetic.wav`로 화면 흐름을 테스트할 수 있습니다.")
    st.stop()

st.audio(uploaded)

st.subheader("1. 파일 기반 STT")
col1, col2 = st.columns([1, 2])

with col1:
    run_stt = st.button("▶ STT 실행", type="primary", use_container_width=True)

with col2:
    st.write("실제 STT는 Google Web Speech API를 호출하므로 인터넷 연결이 필요합니다.")

if run_stt:
    temp_files = []
    try:
        temp_path = save_uploaded_file(uploaded)
        temp_files.append(temp_path)
        if use_demo_text:
            st.session_state["stt_text"] = load_demo_transcript()
            st.session_state["stt_message"] = "오프라인 데모 텍스트를 표시했습니다."
        else:
            result = transcribe_file(temp_path, language=language)
            st.session_state["stt_text"] = result.text
            st.session_state["stt_message"] = result.message
    finally:
        cleanup_temp_files(temp_files)

if "stt_message" in st.session_state:
    if st.session_state.get("stt_text"):
        st.success(st.session_state["stt_message"])
    else:
        st.warning(st.session_state["stt_message"])

if "stt_text" in st.session_state:
    st.text_area("인식 결과", st.session_state["stt_text"], height=180)

st.divider()
st.subheader("2. 화자 분리")

method = st.radio(
    "화자 분리 방식",
    ["KMeans", "pyannote"],
    horizontal=True,
    help="KMeans는 원리 이해용, pyannote는 선택 실습용입니다.",
)
run_diarization = st.button("🔎 화자 분리 실행", use_container_width=True)

if run_diarization:
    temp_files = []
    try:
        temp_path = save_uploaded_file(uploaded)
        temp_files.append(temp_path)

        if method == "KMeans":
            diarization = diarize_kmeans(
                temp_path,
                n_speakers=int(n_speakers),
                chunk_seconds=float(chunk_seconds),
                merge=True,
            )
        else:
            from core.diarize_pyannote import diarize_pyannote

            diarization = diarize_pyannote(temp_path)

        st.session_state["diarization"] = diarization
        st.success("화자 분리 시간표를 만들었습니다.")
    except Exception as exc:
        st.error(f"화자 분리 실패: {exc}")
        with st.expander("오류 상세 보기"):
            st.code(traceback.format_exc())
    finally:
        cleanup_temp_files(temp_files)

if "diarization" in st.session_state:
    diarization = st.session_state["diarization"]
    st.dataframe(diarization, use_container_width=True)
    st.code(format_timeline(diarization), language="text")

st.divider()
st.subheader("3. 화자 구간별 STT로 회의록 표 만들기")

st.write(
    "화자 분리 결과의 각 구간을 다시 잘라 STT에 넣으면 `시간 / 화자 / 문장` 표를 만들 수 있습니다. "
    "긴 오디오는 네트워크 호출이 많아질 수 있으니 짧은 파일로 먼저 테스트하세요."
)

run_pipeline = st.button("🧩 화자 분리 + 구간별 STT 실행", use_container_width=True)

if run_pipeline:
    temp_files = []
    try:
        temp_path = save_uploaded_file(uploaded)
        temp_files.append(temp_path)
        pipeline_method = "kmeans" if method == "KMeans" else "pyannote"
        minutes_df = build_diarized_transcript(
            temp_path,
            method=pipeline_method,
            language=language,
            n_speakers=int(n_speakers),
            chunk_seconds=float(chunk_seconds),
            use_demo_text=use_demo_text,
        )
        st.session_state["minutes_df"] = minutes_df
        st.session_state["minutes_text"] = format_minutes(minutes_df)
        st.success("회의록 표를 만들었습니다.")
    except Exception as exc:
        st.error(f"회의록 생성 실패: {exc}")
        with st.expander("오류 상세 보기"):
            st.code(traceback.format_exc())
    finally:
        cleanup_temp_files(temp_files)

if "minutes_df" in st.session_state:
    st.dataframe(st.session_state["minutes_df"], use_container_width=True)
    st.text_area("회의록 텍스트", st.session_state["minutes_text"], height=240)
