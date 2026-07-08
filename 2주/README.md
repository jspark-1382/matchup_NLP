# 2주차 실습 파일 - 소리에서 회의록까지

이 프로젝트는 2주차 강의자료의 실습 흐름을 그대로 따라가도록 만든 예제 파일 묶음입니다.
핵심 흐름은 **오디오 업로드 → STT → Streamlit 화면 출력 → 화자 분리 → 시간/화자/문장 표 생성**입니다.

## 1. 폴더 구조

```text
week2_ai_meeting_practice/
├── app.py                         # Streamlit 통합 실습 앱
├── requirements.txt               # 기본 실습 패키지
├── requirements_optional.txt      # pyannote 등 선택 패키지
├── .env.example                   # Hugging Face 토큰 예시
├── core/
│   ├── stt.py                     # 음성 파일/마이크 STT 함수
│   ├── diarize.py                 # KMeans 기반 간단 화자 분리
│   ├── diarize_pyannote.py        # pyannote 기반 화자 분리 선택 실습
│   ├── pipeline.py                # 화자 구간 + STT 결합 파이프라인
│   └── audio_utils.py             # 오디오 저장/변환/구간 자르기 유틸
├── exercises/
│   ├── 01_file_stt.py             # 파일 기반 STT 실습
│   ├── 02_microphone_stt.py       # 마이크 입력 STT 실습
│   ├── 03_streamlit_minimal.py    # 최소 Streamlit 앱
│   ├── 04_kmeans_diarize.py       # KMeans 화자 분리 실습
│   ├── 05_pyannote_diarize.py     # pyannote 화자 분리 실습
│   └── 06_diarized_stt_pipeline.py# 화자 분리 + 구간별 STT 결합 실습
├── scripts/
│   └── make_sample_audio.py       # 실습용 합성 오디오 생성 스크립트
└── data/
    ├── sample.wav                 # 업로드/재생 테스트용 짧은 오디오
    ├── meeting_synthetic.wav      # KMeans 화자 분리 테스트용 합성 오디오
    └── sample_transcript.txt      # 오프라인 데모용 회의 문장
```

## 2. 빠른 시작

```bash
cd week2_ai_meeting_practice
python -m venv .venv
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

패키지 설치:

```bash
pip install -r requirements.txt
```

Streamlit 앱 실행:

```bash
streamlit run app.py
```

브라우저에서 `http://localhost:8501` 화면이 열리면 오디오 파일을 업로드하고 `STT 실행` 버튼을 누르면 됩니다.

## 3. 실습 순서

### 실습 1: 파일 기반 STT

```bash
python exercises/01_file_stt.py data/sample.wav
```

`recognize_google(..., language="ko-KR")`를 사용합니다. 실제 한국어 텍스트 인식을 확인하려면 직접 녹음한 한국어 `wav` 파일을 넣어 주세요. 기본 제공 `sample.wav`는 합성 신호라서 STT 정확도 확인용이 아니라 파일 열기와 오류 처리 확인용입니다.

### 실습 2: Streamlit 웹앱

```bash
streamlit run exercises/03_streamlit_minimal.py
```

강의자료의 최소 흐름처럼 `file_uploader`, `st.audio`, `st.button`, `st.text_area`를 연결합니다.

### 실습 3: KMeans 화자 분리

```bash
python exercises/04_kmeans_diarize.py data/meeting_synthetic.wav --speakers 2 --seconds 5
```

오디오를 고정 길이로 자르고, 각 조각에서 `pitch`, `volume`, `zero_crossing_rate`를 뽑은 뒤 KMeans로 비슷한 목소리 조각을 묶습니다.

### 실습 4: pyannote 화자 분리 선택 실습

pyannote는 성능이 좋지만 설치와 토큰 준비가 필요합니다.

```bash
pip install -r requirements_optional.txt
cp .env.example .env
# .env 파일에 HF_TOKEN 값을 입력
python exercises/05_pyannote_diarize.py data/meeting_synthetic.wav
```

## 4. 주의할 점

- Google STT는 인터넷 연결이 필요합니다.
- `speech_recognition.AudioFile`은 기본적으로 `wav`, `aiff`, `flac` 계열이 가장 안정적입니다.
- `mp3`, `m4a`를 넣으면 `pydub`과 `ffmpeg`가 필요할 수 있습니다.
- 마이크 실습은 운영체제의 마이크 권한이 먼저 켜져 있어야 합니다.
- pyannote 실습에서는 Hugging Face 토큰을 코드에 직접 쓰지 말고 `.env`나 환경변수로 넣으세요.

## 5. 강의자료와 연결되는 포인트

- 파일 기반 STT: 오디오 파일을 열고 `Recognizer`가 `record()`로 읽은 뒤 `recognize_google(..., language="ko-KR")`로 변환합니다.
- Streamlit: 콘솔 결과를 사용자가 만지는 웹 화면으로 바꿉니다.
- 화자 분리: STT가 “무슨 말”을 찾는다면, 화자 분리는 “누가 언제 말했는가”를 찾습니다.
- KMeans 방식: 원리 이해용 돋보기입니다. 간단하지만 실제 회의에서는 겹침 발화, 잡음, 비슷한 목소리에 약할 수 있습니다.
- pyannote 방식: 사전 훈련된 화자 분리 파이프라인으로 시간표를 더 정교하게 얻는 선택 실습입니다.
