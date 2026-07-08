"""실습 1: 파일 기반 STT.

사용법:
    python exercises/01_file_stt.py data/sample.wav
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from core.stt import transcribe_file  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="오디오 파일을 한국어 텍스트로 변환합니다.")
    parser.add_argument("audio", type=Path, help="wav/flac/aiff 또는 변환 가능한 오디오 파일 경로")
    parser.add_argument("--language", default="ko-KR", help="STT 언어 코드. 기본값: ko-KR")
    args = parser.parse_args()

    result = transcribe_file(args.audio, language=args.language)
    print(f"상태: {result.message}")
    if result.text:
        print("\n인식 결과")
        print("-" * 40)
        print(result.text)


if __name__ == "__main__":
    main()
