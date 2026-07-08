"""실습용 합성 오디오 생성 스크립트.

실제 사람 음성이 아니라, 업로드/재생/KMeans 화자 분리 흐름을 테스트하기 위한
두 종류의 톤 신호를 만듭니다.
"""

from __future__ import annotations

import csv
import math
import random
import wave
from pathlib import Path

SAMPLE_RATE = 16_000


def sine_mix(frequency: float, t: float, amplitude: float = 0.3) -> float:
    """목소리처럼 보이도록 기본 주파수와 약한 배음을 섞습니다."""
    base = math.sin(2 * math.pi * frequency * t)
    harmonic = 0.35 * math.sin(2 * math.pi * frequency * 2 * t)
    slow = 0.15 * math.sin(2 * math.pi * 3 * t)
    noise = random.uniform(-0.015, 0.015)
    return amplitude * (base + harmonic + slow) + noise


def silence(t: float) -> float:
    """아주 작은 배경 잡음이 있는 침묵."""
    return random.uniform(-0.006, 0.006)


def write_wav(path: Path, samples: list[float], sample_rate: int = SAMPLE_RATE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "w") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        for sample in samples:
            clipped = max(-1.0, min(1.0, sample))
            wav.writeframesraw(int(clipped * 32767).to_bytes(2, byteorder="little", signed=True))


def generate_tone(duration: float, frequency: float, amplitude: float) -> list[float]:
    total = int(duration * SAMPLE_RATE)
    return [sine_mix(frequency, idx / SAMPLE_RATE, amplitude) for idx in range(total)]


def generate_silence(duration: float) -> list[float]:
    total = int(duration * SAMPLE_RATE)
    return [silence(idx / SAMPLE_RATE) for idx in range(total)]


def make_sample_wav(output_dir: Path) -> None:
    samples: list[float] = []
    samples += generate_tone(1.2, 180, 0.24)
    samples += generate_silence(0.4)
    samples += generate_tone(1.2, 260, 0.18)
    samples += generate_silence(0.4)
    samples += generate_tone(1.2, 210, 0.22)
    write_wav(output_dir / "sample.wav", samples)


def make_meeting_synthetic(output_dir: Path) -> None:
    # speaker별 주파수와 크기를 다르게 만들어 KMeans가 구분할 단서를 줍니다.
    plan = [
        (0.0, 5.0, "SPEAKER_00", 165, 0.30),
        (5.0, 10.0, "SPEAKER_01", 255, 0.22),
        (10.0, 15.0, "SPEAKER_00", 170, 0.29),
        (15.0, 20.0, "SPEAKER_01", 250, 0.23),
        (20.0, 25.0, "SPEAKER_00", 160, 0.31),
        (25.0, 30.0, "SPEAKER_01", 262, 0.21),
        (30.0, 35.0, "SPEAKER_00", 168, 0.30),
        (35.0, 40.0, "SPEAKER_01", 248, 0.22),
    ]

    samples: list[float] = []
    rows: list[dict[str, object]] = []
    for start, end, speaker, frequency, amplitude in plan:
        # 각 5초 구간의 앞뒤 0.15초 정도를 작게 만들어 실제 회의의 끊김 느낌을 냅니다.
        duration = end - start
        segment = generate_tone(duration, frequency, amplitude)
        fade_samples = int(0.15 * SAMPLE_RATE)
        for idx in range(min(fade_samples, len(segment))):
            factor = idx / fade_samples
            segment[idx] *= factor
            segment[-idx - 1] *= factor
        samples.extend(segment)
        rows.append({"start": start, "end": end, "speaker": speaker, "note": "synthetic tone"})

    write_wav(output_dir / "meeting_synthetic.wav", samples)

    with (output_dir / "synthetic_labels.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["start", "end", "speaker", "note"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    output_dir = Path(__file__).resolve().parents[1] / "data"
    make_sample_wav(output_dir)
    make_meeting_synthetic(output_dir)
    print(f"생성 완료: {output_dir}")


if __name__ == "__main__":
    main()
