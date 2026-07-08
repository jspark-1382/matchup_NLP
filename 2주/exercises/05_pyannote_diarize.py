"""실습 5: pyannote 기반 화자 분리.

준비:
    pip install -r requirements_optional.txt
    cp .env.example .env
    # .env 파일에 HF_TOKEN 입력

사용법:
    python exercises/05_pyannote_diarize.py data/meeting_synthetic.wav
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from core.diarize_pyannote import diarize_pyannote  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="pyannote 파이프라인으로 화자 시간표를 추출합니다.")
    parser.add_argument("audio", type=Path, help="오디오 파일 경로")
    parser.add_argument("--csv", type=Path, default=None, help="결과를 저장할 CSV 경로")
    args = parser.parse_args()

    df = diarize_pyannote(args.audio)
    print(df.to_string(index=False))

    if args.csv:
        df.to_csv(args.csv, index=False, encoding="utf-8-sig")
        print(f"CSV 저장 완료: {args.csv}")


if __name__ == "__main__":
    main()
