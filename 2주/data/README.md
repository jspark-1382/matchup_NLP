# data 폴더

- `sample.wav`: 업로드와 재생, 오류 처리 흐름을 확인하기 위한 짧은 합성 오디오입니다.
- `meeting_synthetic.wav`: 두 종류의 톤이 번갈아 나오는 KMeans 화자 분리 테스트용 합성 오디오입니다.
- `synthetic_labels.csv`: `meeting_synthetic.wav`의 의도된 화자 구간입니다.
- `sample_transcript.txt`: 인터넷이 없을 때 화면 흐름을 확인하는 오프라인 데모 문장입니다.

합성 오디오는 실제 사람 음성이 아니므로 Google STT 결과가 정확하지 않습니다. STT 실습은 직접 녹음한 한국어 `wav` 파일로 확인하는 것이 좋습니다.
