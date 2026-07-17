# AI 회의록 Subagent 실습 — 수강생용 Starter

완성된 STT·화자 분리 Demo Adapter와 Fixture를 이용해 세 Subagent가 회의록 웹 서비스를 완성하는 프로젝트입니다.

## 이미 제공된 것

- `core/transcribe.py`: Demo Mode 대화록 Adapter
- `fixtures/sample_transcript.json`: 화자별 대화록
- `fixtures/sample_gemini_summary.json`: Gemini Mock 응답
- `fixtures/sample_meeting_demo.wav`: 업로드·재생 확인용 안내음
- `tests/`: 완성 조건을 확인하는 테스트

## 여러분이 완성할 것

1. `.opencode/agents/`에 세 Subagent 파일 만들기
2. `app.py`, `components/` Streamlit 화면 완성
3. `core/gemini_summary.py` 구조화 요약 완성
4. `core/traditional_summary.py` TF-IDF·TextRank 요약 완성
5. 두 요약 결과 통합과 테스트

## 시작

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## 첫 상태

Starter의 요약 함수는 빈 결과를 반환하므로 일부 테스트가 실패하는 것이 정상입니다. 각 Agent 구현이 끝날 때마다 다음을 실행합니다.

```bash
python -m unittest discover -s tests -v
```

최종적으로 모든 테스트가 통과해야 합니다. API 키가 없어도 Mock과 Demo Mode로 완주할 수 있습니다.

