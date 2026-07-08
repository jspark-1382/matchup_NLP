"""실습 4: KMeans 기반 화자 분리.

사용법:
    python exercises/04_kmeans_diarize.py data/meeting_synthetic.wav --speakers 2 --seconds 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from core.diarize import diarize_kmeans, format_timeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="KMeans로 비슷한 목소리 조각을 묶습니다.")
    parser.add_argument("audio", type=Path, help="오디오 파일 경로")
    parser.add_argument("--speakers", type=int, default=2, help="예상 화자 수")
    parser.add_argument("--seconds", type=float, default=5.0, help="오디오 자르기 간격")
    parser.add_argument("--csv", type=Path, default=None, help="결과를 저장할 CSV 경로")
    args = parser.parse_args()

    df = diarize_kmeans(
        args.audio,
        n_speakers=args.speakers,
        chunk_seconds=args.seconds,
        merge=True,
    )

    print("화자 분리 결과")
    print("-" * 40)
    print(format_timeline(df))
    print("\n표")
    print(df.to_string(index=False))

    if args.csv:
        df.to_csv(args.csv, index=False, encoding="utf-8-sig")
        print(f"\nCSV 저장 완료: {args.csv}")


if __name__ == "__main__":
    main()
