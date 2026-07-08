"""KMeans 기반 간단 화자 분리 실습.

이 파일은 성능보다 원리 이해를 목표로 합니다.
오디오를 일정 길이로 자른 뒤 각 조각에서 pitch, volume 같은 단순 특징을 뽑고
KMeans로 비슷한 조각끼리 묶습니다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


@dataclass
class AudioChunk:
    """오디오 조각 정보."""

    start: float
    end: float
    samples: np.ndarray


def load_audio(audio_path: str | Path, target_sr: int | None = 16000) -> tuple[np.ndarray, int]:
    """오디오를 mono waveform으로 불러옵니다."""
    y, sr = librosa.load(str(audio_path), sr=target_sr, mono=True)
    return y, sr


def split_fixed_chunks(
    y: np.ndarray,
    sr: int,
    chunk_seconds: float = 5.0,
    min_seconds: float = 1.0,
) -> list[AudioChunk]:
    """오디오를 고정 길이 조각으로 자릅니다."""
    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds는 0보다 커야 합니다.")

    chunk_size = int(chunk_seconds * sr)
    min_size = int(min_seconds * sr)
    chunks: list[AudioChunk] = []

    for start_sample in range(0, len(y), chunk_size):
        end_sample = min(start_sample + chunk_size, len(y))
        samples = y[start_sample:end_sample]
        if len(samples) < min_size:
            continue
        chunks.append(
            AudioChunk(
                start=start_sample / sr,
                end=end_sample / sr,
                samples=samples,
            )
        )
    return chunks


def _safe_pitch(samples: np.ndarray, sr: int) -> float:
    """대략적인 평균 음높이를 계산합니다."""
    if len(samples) < int(sr * 0.1):
        return 0.0
    try:
        f0 = librosa.yin(samples, fmin=70, fmax=500, sr=sr)
        valid = f0[np.isfinite(f0)]
        if len(valid) == 0:
            return 0.0
        return float(np.nanmedian(valid))
    except Exception:
        return 0.0


def extract_features(chunks: list[AudioChunk], sr: int, min_rms: float = 0.001) -> pd.DataFrame:
    """각 오디오 조각에서 단순 목소리 특징을 추출합니다.

    pitch: 평균 음높이
    volume: 평균 에너지
    zcr: 파형이 0을 지나는 빈도
    spectral_centroid: 소리의 밝기 정도
    """
    rows: list[dict[str, float]] = []

    for idx, chunk in enumerate(chunks):
        samples = chunk.samples.astype(np.float32)
        volume = float(np.sqrt(np.mean(np.square(samples))))
        if volume < min_rms:
            continue

        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=samples)))
        centroid = float(np.mean(librosa.feature.spectral_centroid(y=samples, sr=sr)))
        pitch = _safe_pitch(samples, sr)

        rows.append(
            {
                "chunk_id": idx,
                "start": chunk.start,
                "end": chunk.end,
                "pitch": pitch,
                "volume": volume,
                "zcr": zcr,
                "spectral_centroid": centroid,
            }
        )

    return pd.DataFrame(rows)


def _map_cluster_to_speaker(features: pd.DataFrame, labels: np.ndarray) -> list[str]:
    """KMeans의 임의 cluster 번호를 SPEAKER_00, SPEAKER_01 형태로 정리합니다."""
    temp = features.copy()
    temp["raw_cluster"] = labels
    order = (
        temp.groupby("raw_cluster")[["pitch", "volume"]]
        .mean()
        .sort_values(["pitch", "volume"])
        .index.tolist()
    )
    mapping = {cluster_id: f"SPEAKER_{rank:02d}" for rank, cluster_id in enumerate(order)}
    return [mapping[label] for label in labels]


def merge_adjacent_segments(df: pd.DataFrame, max_gap: float = 0.15) -> pd.DataFrame:
    """연속된 같은 화자 구간을 하나로 합칩니다."""
    if df.empty:
        return df.copy()

    sorted_df = df.sort_values("start").reset_index(drop=True)
    merged: list[dict[str, object]] = []

    current = sorted_df.iloc[0].to_dict()
    for _, row in sorted_df.iloc[1:].iterrows():
        same_speaker = row["speaker"] == current["speaker"]
        close_enough = float(row["start"]) - float(current["end"]) <= max_gap
        if same_speaker and close_enough:
            current["end"] = float(row["end"])
            for col in ["pitch", "volume", "zcr", "spectral_centroid"]:
                current[col] = float(np.nanmean([current[col], row[col]]))
        else:
            merged.append(current)
            current = row.to_dict()
    merged.append(current)

    return pd.DataFrame(merged)


def diarize_kmeans(
    audio_path: str | Path,
    n_speakers: int = 2,
    chunk_seconds: float = 5.0,
    target_sr: int | None = 16000,
    min_rms: float = 0.001,
    merge: bool = True,
) -> pd.DataFrame:
    """KMeans 방식으로 화자 구간을 추정합니다.

    반환 컬럼: start, end, speaker, pitch, volume, zcr, spectral_centroid, raw_cluster
    """
    if n_speakers < 1:
        raise ValueError("n_speakers는 1 이상이어야 합니다.")

    y, sr = load_audio(audio_path, target_sr=target_sr)
    chunks = split_fixed_chunks(y, sr, chunk_seconds=chunk_seconds)
    features = extract_features(chunks, sr, min_rms=min_rms)

    if features.empty:
        raise ValueError("분석할 수 있는 말소리 조각이 없습니다. 녹음 상태나 min_rms 값을 확인해 주세요.")

    if len(features) < n_speakers:
        raise ValueError(
            f"분석 가능한 조각 수({len(features)})가 화자 수({n_speakers})보다 적습니다. "
            "chunk_seconds를 줄이거나 n_speakers를 줄여 주세요."
        )

    feature_columns = ["pitch", "volume", "zcr", "spectral_centroid"]
    scaled = StandardScaler().fit_transform(features[feature_columns])
    model = KMeans(n_clusters=n_speakers, random_state=42, n_init=10)
    labels = model.fit_predict(scaled)

    result = features.copy()
    result["raw_cluster"] = labels
    result["speaker"] = _map_cluster_to_speaker(features, labels)
    result = result[
        [
            "start",
            "end",
            "speaker",
            "pitch",
            "volume",
            "zcr",
            "spectral_centroid",
            "raw_cluster",
        ]
    ]

    if merge:
        result = merge_adjacent_segments(result)

    return result.sort_values("start").reset_index(drop=True)


def format_timeline(df: pd.DataFrame) -> str:
    """화자 분리 결과를 사람이 읽기 좋은 텍스트로 바꿉니다."""
    lines: list[str] = []
    for _, row in df.iterrows():
        lines.append(f"{row['start']:.1f}s - {row['end']:.1f}s  {row['speaker']}")
    return "\n".join(lines)
