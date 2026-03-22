#!/usr/bin/env python3
import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional


@dataclass
class OcrSegment:
    start: float
    end: float
    text: str
    normalized: str


TIME_LINE_RE = re.compile(r"^\d{1,2}:\d{2}\s*(?:AM|PM)?$", re.IGNORECASE)
TIME_TOKEN_RE = re.compile(r"\b\d{1,2}:\d{2}\s*(?:AM|PM)?\b", re.IGNORECASE)


def parse_args():
    parser = argparse.ArgumentParser(
        description="비디오 프레임에서 화면 문자열을 OCR로 추출합니다."
    )
    parser.add_argument("input", type=Path, help="입력 비디오 파일 경로 (mov/mp4 등)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="결과 파일 저장 디렉터리 (기본: 현재 디렉터리)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="프레임 샘플링 간격(초) (기본: 1.0)",
    )
    parser.add_argument(
        "--lang",
        default="eng",
        help="Tesseract 언어 코드 (예: eng, eng+kor). 기본: eng",
    )
    parser.add_argument(
        "--psm",
        type=int,
        default=6,
        help="Tesseract page segmentation mode (기본: 6)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=2.0,
        help="OCR 전에 프레임 확대 배율 (기본: 2.0)",
    )
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=0.6,
        help="인접 프레임을 같은 텍스트로 합칠 유사도 임계값 (기본: 0.6)",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=4,
        help="유효 텍스트로 간주할 최소 글자 수(공백 제외) (기본: 4)",
    )
    parser.add_argument(
        "--keep-frames",
        action="store_true",
        help="OCR용 추출 프레임 PNG를 보존합니다.",
    )
    parser.add_argument(
        "--skip-postprocess",
        action="store_true",
        help="후처리(cleaned 출력/GPT 프롬프트 생성)를 건너뜁니다.",
    )
    parser.add_argument(
        "--ui-repeat-threshold",
        type=float,
        default=0.7,
        help="상단/하단 UI 줄을 제거할 반복 비율 임계값 (기본: 0.7)",
    )
    parser.add_argument(
        "--postprocess-merge-similarity",
        type=float,
        default=0.55,
        help="후처리 후 인접 세그먼트를 다시 합칠 유사도 기준 (기본: 0.55)",
    )
    return parser.parse_args()


def ensure_binary(name: str) -> None:
    if shutil.which(name):
        return
    raise RuntimeError(f"{name}를 찾을 수 없습니다. 먼저 설치하세요.")


def get_tesseract_languages() -> set[str]:
    try:
        result = subprocess.run(
            ["tesseract", "--list-langs"],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("tesseract를 찾을 수 없습니다. 먼저 설치하세요.") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip()
        raise RuntimeError(f"tesseract 언어 목록 조회 실패:\n{stderr}") from exc

    langs = set()
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("List of available languages"):
            continue
        langs.add(line)
    return langs


def validate_languages(lang_arg: str, installed_langs: set[str]) -> None:
    requested = {part.strip() for part in lang_arg.split("+") if part.strip()}
    missing = sorted(requested - installed_langs)
    if not missing:
        return
    installed = ", ".join(sorted(installed_langs)) or "(없음)"
    missing_text = ", ".join(missing)
    raise RuntimeError(
        f"Tesseract 언어 데이터가 없습니다: {missing_text}\n"
        f"현재 설치된 언어: {installed}"
    )


def probe_duration(video_path: Path) -> Optional[float]:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    text = result.stdout.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def build_filter(interval: float, scale: float) -> str:
    filters = [f"fps=1/{interval}"]
    if scale != 1.0:
        filters.append(
            f"scale=trunc(iw*{scale}/2)*2:trunc(ih*{scale}/2)*2:flags=lanczos"
        )
    filters.append("format=gray")
    return ",".join(filters)


def extract_frames(video_path: Path, frames_dir: Path, interval: float, scale: float) -> list[Path]:
    filter_expr = build_filter(interval, scale)
    output_pattern = frames_dir / "frame_%06d.png"
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(video_path),
                "-vf",
                filter_expr,
                str(output_pattern),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg를 찾을 수 없습니다. 먼저 설치하세요.") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip()
        raise RuntimeError(f"프레임 추출 실패:\n{stderr}") from exc

    frames = sorted(frames_dir.glob("frame_*.png"))
    if not frames:
        raise RuntimeError("OCR용 프레임이 생성되지 않았습니다.")
    return frames


def clean_ocr_text(text: str) -> str:
    lines = []
    for line in text.splitlines():
        normalized = " ".join(line.split()).strip(" |-_")
        if sum(ch.isalnum() for ch in normalized) >= 3:
            lines.append(normalized)
    return "\n".join(lines)


def normalize_for_compare(text: str) -> str:
    return "".join(ch.casefold() for ch in text if ch.isalnum())


def comparison_key(text: str) -> str:
    seen = set()
    keys = []
    for line in text.splitlines():
        normalized = normalize_for_compare(line)
        if len(normalized) < 6 or normalized in seen:
            continue
        seen.add(normalized)
        keys.append(normalized)
    if keys:
        return "\n".join(keys)
    return normalize_for_compare(text)


def is_useful_text(text: str, min_chars: int) -> bool:
    compact = "".join(ch for ch in text if not ch.isspace())
    if len(compact) < min_chars:
        return False
    return any(ch.isalnum() for ch in compact)


def text_quality(text: str) -> tuple[int, int]:
    return (len(comparison_key(text)), len(text))


def ocr_frame(frame_path: Path, lang: str, psm: int) -> str:
    try:
        result = subprocess.run(
            [
                "tesseract",
                str(frame_path),
                "stdout",
                "-l",
                lang,
                "--psm",
                str(psm),
                "quiet",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("tesseract를 찾을 수 없습니다. 먼저 설치하세요.") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip()
        raise RuntimeError(f"OCR 실패 ({frame_path.name}):\n{stderr}") from exc
    return clean_ocr_text(result.stdout)


def to_srt_timestamp(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h = ms // 3_600_000
    ms %= 3_600_000
    m = ms // 60_000
    ms %= 60_000
    s = ms // 1000
    ms %= 1000
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def to_readable_timestamp(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h = ms // 3_600_000
    ms %= 3_600_000
    m = ms // 60_000
    ms %= 60_000
    s = ms // 1000
    ms %= 1000
    return f"{h:02}:{m:02}:{s:02}.{ms:03}"


def similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    sequence_ratio = SequenceMatcher(None, left, right).ratio()
    left_lines = {line for line in left.splitlines() if line}
    right_lines = {line for line in right.splitlines() if line}
    if not left_lines or not right_lines:
        return sequence_ratio
    overlap_ratio = len(left_lines & right_lines) / min(len(left_lines), len(right_lines))
    return max(sequence_ratio, overlap_ratio)


def split_lines(text: str) -> list[str]:
    lines = []
    for line in text.splitlines():
        collapsed = " ".join(line.split()).strip(" |-_~")
        if collapsed:
            lines.append(collapsed)
    return lines


def line_is_low_signal(line: str) -> bool:
    normalized = normalize_for_compare(line)
    if len(normalized) < 4:
        return True
    if TIME_LINE_RE.match(line):
        return True
    symbol_count = sum(1 for ch in line if not ch.isalnum() and not ch.isspace())
    if TIME_TOKEN_RE.search(line) and len(normalized) <= 10:
        return True
    if line and symbol_count / len(line) > 0.35 and len(normalized) < 10:
        return True
    return False


def line_looks_like_ui_chrome(line: str, index: int, total_lines: int) -> bool:
    normalized = normalize_for_compare(line)
    if not normalized:
        return True

    near_edge = index < 3 or index >= max(total_lines - 3, 0)
    if not near_edge:
        return False

    symbol_count = sum(1 for ch in line if not ch.isalnum() and not ch.isspace())
    symbol_ratio = symbol_count / len(line) if line else 0.0

    if TIME_TOKEN_RE.search(line) and len(normalized) <= 16:
        return True
    if symbol_ratio > 0.18 and len(normalized) <= 24:
        return True
    if any(token in line for token in ("<", ">", "@", "\\", "[", "]")) and len(normalized) <= 24:
        return True
    return False


def detect_ui_chrome(segments: list[OcrSegment], repeat_threshold: float) -> set[str]:
    top_counts: dict[str, int] = {}
    bottom_counts: dict[str, int] = {}
    total = max(len(segments), 1)

    for segment in segments:
        lines = split_lines(segment.text)
        for line in lines[:3]:
            key = normalize_for_compare(line)
            if 4 <= len(key) <= 20:
                top_counts[key] = top_counts.get(key, 0) + 1
        for line in lines[-3:]:
            key = normalize_for_compare(line)
            if 4 <= len(key) <= 20:
                bottom_counts[key] = bottom_counts.get(key, 0) + 1

    chrome = set()
    for key, count in top_counts.items():
        if count / total >= repeat_threshold:
            chrome.add(key)
    for key, count in bottom_counts.items():
        if count / total >= repeat_threshold:
            chrome.add(key)
    return chrome


def clean_segment_text(text: str, ui_chrome: set[str]) -> str:
    cleaned = []
    seen = set()
    lines = split_lines(text)
    total_lines = len(lines)
    for index, line in enumerate(lines):
        key = normalize_for_compare(line)
        if not key or key in seen:
            continue
        if key in ui_chrome or line_is_low_signal(line) or line_looks_like_ui_chrome(line, index, total_lines):
            continue
        seen.add(key)
        cleaned.append(line)
    return "\n".join(cleaned)


def postprocess_segments(
    segments: list[OcrSegment],
    ui_repeat_threshold: float,
    merge_similarity: float,
) -> list[OcrSegment]:
    ui_chrome = detect_ui_chrome(segments, repeat_threshold=ui_repeat_threshold)
    cleaned_segments: list[OcrSegment] = []

    for segment in segments:
        cleaned_text = clean_segment_text(segment.text, ui_chrome=ui_chrome)
        if not cleaned_text:
            continue

        normalized = comparison_key(cleaned_text)
        updated = OcrSegment(
            start=segment.start,
            end=segment.end,
            text=cleaned_text,
            normalized=normalized,
        )

        if cleaned_segments:
            prev = cleaned_segments[-1]
            gap = updated.start - prev.end
            max_gap = max(1.5, updated.end - updated.start)
            score = similarity(prev.normalized, updated.normalized)
            if gap <= max_gap and score >= merge_similarity:
                prev.end = updated.end
                if text_quality(updated.text) > text_quality(prev.text):
                    prev.text = updated.text
                    prev.normalized = updated.normalized
                continue

        cleaned_segments.append(updated)

    return cleaned_segments


def merge_segments(
    frames: list[Path],
    interval: float,
    lang: str,
    psm: int,
    min_chars: int,
    min_similarity: float,
    duration: Optional[float],
) -> list[OcrSegment]:
    segments: list[OcrSegment] = []
    total = len(frames)

    for index, frame_path in enumerate(frames):
        if total <= 10 or index % 10 == 0 or index == total - 1:
            print(f"  OCR 진행: {index + 1}/{total}")

        raw_text = ocr_frame(frame_path, lang=lang, psm=psm)
        if not is_useful_text(raw_text, min_chars=min_chars):
            continue

        normalized = comparison_key(raw_text)
        start = index * interval
        end = start + interval
        if duration is not None:
            end = min(end, duration)

        if segments:
            prev = segments[-1]
            gap = start - prev.end
            score = similarity(prev.normalized, normalized)
            if gap <= interval * 1.5 and score >= min_similarity:
                prev.end = end
                if text_quality(raw_text) > text_quality(prev.text):
                    prev.text = raw_text
                    prev.normalized = normalized
                continue

        segments.append(OcrSegment(start=start, end=end, text=raw_text, normalized=normalized))

    return segments


def write_txt(segments: list[OcrSegment], out_path: Path) -> None:
    blocks = [segment.text for segment in segments if segment.text]
    payload = "\n\n".join(blocks)
    if payload:
        payload += "\n"
    out_path.write_text(payload, encoding="utf-8")


def write_timed_txt(segments: list[OcrSegment], out_path: Path) -> None:
    blocks = []
    for segment in segments:
        if not segment.text:
            continue
        blocks.append(
            f"[{to_readable_timestamp(segment.start)} - {to_readable_timestamp(segment.end)}]\n{segment.text}"
        )
    payload = "\n\n".join(blocks)
    if payload:
        payload += "\n"
    out_path.write_text(payload, encoding="utf-8")


def write_srt(segments: list[OcrSegment], out_path: Path) -> None:
    blocks = []
    for index, segment in enumerate(segments, start=1):
        if not segment.text:
            continue
        blocks.append(
            f"{index}\n"
            f"{to_srt_timestamp(segment.start)} --> {to_srt_timestamp(segment.end)}\n"
            f"{segment.text}\n"
        )
    payload = "\n".join(blocks)
    if payload:
        payload += "\n"
    out_path.write_text(payload, encoding="utf-8")


def build_timed_text_payload(segments: list[OcrSegment]) -> str:
    blocks = []
    for segment in segments:
        if not segment.text:
            continue
        blocks.append(
            f"[{to_readable_timestamp(segment.start)} - {to_readable_timestamp(segment.end)}]\n{segment.text}"
        )
    payload = "\n\n".join(blocks)
    if payload:
        payload += "\n"
    return payload


def write_gpt_prompt(segments: list[OcrSegment], out_path: Path) -> None:
    timed_payload = build_timed_text_payload(segments).strip()
    prompt = (
        "# OCR Cleanup Prompt\n\n"
        "Use the OCR timeline below to reconstruct the original English on-screen text.\n\n"
        "Rules:\n"
        "- Remove UI chrome, timestamps, and OCR garbage.\n"
        "- Preserve chronology.\n"
        "- Merge overlapping adjacent blocks when they clearly show the same screen.\n"
        "- Fix OCR mistakes only when highly confident.\n"
        "- If uncertain, keep the original wording conservative.\n\n"
        "Output format:\n"
        "1. cleaned_plain_text\n"
        "2. cleaned_timeline\n"
        "3. uncertain_readings\n\n"
        "OCR input:\n\n"
        "```text\n"
        f"{timed_payload}\n"
        "```\n"
    )
    out_path.write_text(prompt, encoding="utf-8")


def next_available_dir(base_dir: Path) -> Path:
    if not base_dir.exists():
        return base_dir
    for suffix in range(1, 1000):
        candidate = base_dir.with_name(f"{base_dir.name}_{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"프레임 디렉터리를 만들 수 없습니다: {base_dir}")


def main() -> int:
    args = parse_args()
    if args.interval <= 0:
        print("--interval 은 0보다 커야 합니다.", file=sys.stderr)
        return 1
    if args.scale <= 0:
        print("--scale 은 0보다 커야 합니다.", file=sys.stderr)
        return 1
    if not (0.0 <= args.min_similarity <= 1.0):
        print("--min-similarity 는 0과 1 사이여야 합니다.", file=sys.stderr)
        return 1
    if not (0.0 <= args.ui_repeat_threshold <= 1.0):
        print("--ui-repeat-threshold 는 0과 1 사이여야 합니다.", file=sys.stderr)
        return 1
    if not (0.0 <= args.postprocess_merge_similarity <= 1.0):
        print("--postprocess-merge-similarity 는 0과 1 사이여야 합니다.", file=sys.stderr)
        return 1

    input_path = args.input.resolve()
    if not input_path.exists():
        print(f"입력 파일이 없습니다: {input_path}", file=sys.stderr)
        return 1

    try:
        ensure_binary("ffmpeg")
        ensure_binary("ffprobe")
        ensure_binary("tesseract")
        installed_langs = get_tesseract_languages()
        validate_languages(args.lang, installed_langs)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem
    txt_path = output_dir / f"{stem}.ocr.txt"
    timed_txt_path = output_dir / f"{stem}.ocr.timed.txt"
    srt_path = output_dir / f"{stem}.ocr.srt"
    cleaned_txt_path = output_dir / f"{stem}.ocr.cleaned.txt"
    cleaned_timed_txt_path = output_dir / f"{stem}.ocr.cleaned.timed.txt"
    cleaned_srt_path = output_dir / f"{stem}.ocr.cleaned.srt"
    gpt_prompt_path = output_dir / f"{stem}.ocr.gpt_prompt.md"

    duration = probe_duration(input_path)

    if args.keep_frames:
        frames_dir = next_available_dir(output_dir / f"{stem}_ocr_frames")
        frames_dir.mkdir(parents=True, exist_ok=False)
        temp_dir_ctx = None
    else:
        temp_dir_ctx = tempfile.TemporaryDirectory(prefix="video_ocr_")
        frames_dir = Path(temp_dir_ctx.name)

    try:
        try:
            print(f"[1/4] 프레임 추출: interval={args.interval}s, scale={args.scale}x")
            frames = extract_frames(input_path, frames_dir=frames_dir, interval=args.interval, scale=args.scale)

            print(f"[2/4] OCR 수행: frames={len(frames)}, lang={args.lang}, psm={args.psm}")
            segments = merge_segments(
                frames=frames,
                interval=args.interval,
                lang=args.lang,
                psm=args.psm,
                min_chars=args.min_chars,
                min_similarity=args.min_similarity,
                duration=duration,
            )

            print(f"[3/4] 원본 OCR 저장: segments={len(segments)}")
            write_txt(segments, txt_path)
            write_timed_txt(segments, timed_txt_path)
            write_srt(segments, srt_path)

            if args.skip_postprocess:
                print("[4/4] 후처리 건너뜀")
            else:
                cleaned_segments = postprocess_segments(
                    segments,
                    ui_repeat_threshold=args.ui_repeat_threshold,
                    merge_similarity=args.postprocess_merge_similarity,
                )
                print(f"[4/4] 후처리 저장: cleaned_segments={len(cleaned_segments)}")
                write_txt(cleaned_segments, cleaned_txt_path)
                write_timed_txt(cleaned_segments, cleaned_timed_txt_path)
                write_srt(cleaned_segments, cleaned_srt_path)
                write_gpt_prompt(cleaned_segments, gpt_prompt_path)
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 1
    finally:
        if temp_dir_ctx is not None:
            temp_dir_ctx.cleanup()

    print(f"- TXT: {txt_path}")
    print(f"- Timed TXT: {timed_txt_path}")
    print(f"- SRT: {srt_path}")
    if not args.skip_postprocess:
        print(f"- Cleaned TXT: {cleaned_txt_path}")
        print(f"- Cleaned Timed TXT: {cleaned_timed_txt_path}")
        print(f"- Cleaned SRT: {cleaned_srt_path}")
        print(f"- GPT Prompt: {gpt_prompt_path}")
    if args.keep_frames:
        print(f"- Frames: {frames_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
