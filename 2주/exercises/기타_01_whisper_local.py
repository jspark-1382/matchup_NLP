"""기타 실습 1: 로컬 Whisper 모델 기반 STT.

사용법:
    python exercises/기타_01_whisper_local.py data/sample.wav
    python exercises/기타_01_whisper_local.py data/meeting_synthetic.wav --model base

설명:
    faster-whisper를 사용하며 models/whisper/ 폴더에 저장된
    로컬 모델을 토큰 없이 로드합니다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from core.whisper_local import transcribe_whisper  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="로컬 Whisper 모델로 오디오를 텍스트로 변환합니다.")
    parser.add_argument("audio", type=Path, help="wav/flac/mp3 등 변환 가능한 오디오 파일 경로")
    parser.add_argument("--model", default="base", help="모델 크기: tiny/base/small/medium/large-v3")
    parser.add_argument("--language", default="ko", help="STT 언어 코드. 기본값: ko")
    args = parser.parse_args()

    print(f"모델 로드 중: {args.model}")
    df = transcribe_whisper(
        args.audio,
        model_size=args.model,
        language=args.language,
    )

    print("\n인식 결과")
    print("-" * 50)
    for _, row in df.iterrows():
        print(f"[{row['start']:.1f}s - {row['end']:.1f}s] {row['text']}")


if __name__ == "__main__":
    main()
