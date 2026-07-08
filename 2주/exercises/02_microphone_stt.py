"""실습 2: 마이크 입력 STT.

운영체제의 마이크 권한과 PyAudio 설치가 필요할 수 있습니다.
사용법:
    python exercises/02_microphone_stt.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from core.stt import transcribe_microphone  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="마이크로 말한 내용을 텍스트로 변환합니다.")
    parser.add_argument("--language", default="ko-KR", help="STT 언어 코드. 기본값: ko-KR")
    parser.add_argument("--seconds", type=int, default=8, help="최대 녹음 시간. 기본값: 8초")
    args = parser.parse_args()

    print("마이크 주변 소음을 보정한 뒤 듣기 시작합니다. 문장을 또렷하게 말해 주세요.")
    result = transcribe_microphone(language=args.language, phrase_time_limit=args.seconds)
    print(f"상태: {result.message}")
    if result.text:
        print("\n인식 결과")
        print("-" * 40)
        print(result.text)


if __name__ == "__main__":
    main()
