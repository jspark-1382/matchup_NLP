"""실습 6: 화자 분리 + 구간별 STT 결합.

사용법:
    python exercises/06_diarized_stt_pipeline.py data/meeting_synthetic.wav --demo
    python exercises/06_diarized_stt_pipeline.py data/S00007846.wav --method kmeans
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from core.pipeline import build_diarized_transcript, format_minutes  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="시간/화자/문장 회의록 표를 만듭니다.")
    parser.add_argument("audio", type=Path, help="오디오 파일 경로")
    parser.add_argument("--method", choices=["kmeans", "pyannote"], default="kmeans")
    parser.add_argument("--speakers", type=int, default=2, help="KMeans 예상 화자 수")
    parser.add_argument("--seconds", type=float, default=5.0, help="KMeans 자르기 간격")
    parser.add_argument("--language", default="ko-KR", help="STT 언어 코드")
    parser.add_argument("--demo", action="store_true", help="실제 STT 대신 데모 문장을 붙입니다.")
    parser.add_argument("--csv", type=Path, default=None, help="결과 CSV 저장 경로")
    args = parser.parse_args()

    df = build_diarized_transcript(
        audio_path=args.audio,
        method=args.method,
        language=args.language,
        n_speakers=args.speakers,
        chunk_seconds=args.seconds,
        use_demo_text=args.demo,
    )

    print("회의록")
    print("-" * 40)
    print(format_minutes(df))
    print("\n표")
    print(df.to_string(index=False))

    if args.csv:
        df.to_csv(args.csv, index=False, encoding="utf-8-sig")
        print(f"\nCSV 저장 완료: {args.csv}")


if __name__ == "__main__":
    main()
