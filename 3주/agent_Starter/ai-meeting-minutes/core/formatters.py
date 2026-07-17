from __future__ import annotations


def format_seconds(value: float) -> str:
    total = max(0, int(round(value)))
    minutes, seconds = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def transcript_to_text(transcript: list[dict]) -> str:
    return "\n".join(
        f"[{format_seconds(item['start'])}-{format_seconds(item['end'])}] "
        f"{item['speaker']}: {item['text'].strip()}"
        for item in transcript
        if item.get("text", "").strip()
    )

