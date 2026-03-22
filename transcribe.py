#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path


def load_whisper_model_class():
    try:
        from faster_whisper import WhisperModel
    except ModuleNotFoundError as exc:
        if exc.name == "faster_whisper":
            print(
                "faster_whisper 패키지를 찾을 수 없습니다.\n"
                "가상환경을 활성화한 뒤 다시 실행하세요:\n"
                "  source .venv/bin/activate\n"
                "  python transcribe.py input.MOV\n\n"
                "또는 가상환경 Python을 직접 사용하세요:\n"
                "  .venv/bin/python transcribe.py input.MOV\n",
                file=sys.stderr,
            )
            raise SystemExit(1)
        raise
    return WhisperModel


def ffmpeg_extract_audio(video_path: Path, wav_path: Path) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(wav_path),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except FileNotFoundError:
        raise RuntimeError("ffmpeg를 찾을 수 없습니다. 먼저 ffmpeg를 설치하세요.")
    except subprocess.CalledProcessError as exc:
        msg = exc.stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(f"ffmpeg 음성 추출 실패:\n{msg}")


def to_srt_timestamp(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h = ms // 3_600_000
    ms %= 3_600_000
    m = ms // 60_000
    ms %= 60_000
    s = ms // 1000
    ms %= 1000
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def to_vtt_timestamp(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h = ms // 3_600_000
    ms %= 3_600_000
    m = ms // 60_000
    ms %= 60_000
    s = ms // 1000
    ms %= 1000
    return f"{h:02}:{m:02}:{s:02}.{ms:03}"


def write_txt(segments, out_path: Path) -> None:
    lines = [seg.text.strip() for seg in segments if seg.text.strip()]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_timestamped_txt(segments, out_path: Path) -> None:
    lines = []
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        lines.append(f"[{seg.start:.2f} - {seg.end:.2f}] {text}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_srt(segments, out_path: Path) -> None:
    blocks = []
    idx = 1
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        start = to_srt_timestamp(seg.start)
        end = to_srt_timestamp(seg.end)
        blocks.append(f"{idx}\n{start} --> {end}\n{text}\n")
        idx += 1
    out_path.write_text("\n".join(blocks) + ("\n" if blocks else ""), encoding="utf-8")


def write_vtt(segments, out_path: Path) -> None:
    lines = ["WEBVTT", ""]
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        start = to_vtt_timestamp(seg.start)
        end = to_vtt_timestamp(seg.end)
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(
        description="비디오 파일에서 음성을 추출해 Whisper로 자막/텍스트를 생성합니다."
    )
    parser.add_argument("input", type=Path, help="입력 비디오 파일 경로 (mov/mp4 등)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="결과 파일 저장 디렉터리 (기본: 현재 디렉터리)",
    )
    parser.add_argument(
        "--model-size",
        default="small",
        choices=["tiny", "base", "small", "medium", "large-v3", "distil-large-v3"],
        help="Whisper 모델 크기 (기본: small)",
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"], help="실행 디바이스")
    parser.add_argument(
        "--compute-type",
        default="int8",
        help="연산 타입 (예: int8, int8_float16, float16, float32)",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="언어 코드 지정(예: ko, en). 미지정 시 자동 감지",
    )
    parser.add_argument(
        "--task",
        default="transcribe",
        choices=["transcribe", "translate"],
        help="transcribe(원문) 또는 translate(영문 번역)",
    )
    parser.add_argument(
        "--keep-audio",
        action="store_true",
        help="중간 wav 파일 보존 (기본: 작업 후 삭제)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    in_path = args.input.resolve()
    if not in_path.exists():
        print(f"입력 파일이 없습니다: {in_path}", file=sys.stderr)
        return 1

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = in_path.stem

    wav_path = out_dir / f"{stem}.wav"
    txt_path = out_dir / f"{stem}.txt"
    timed_txt_path = out_dir / f"{stem}.timed.txt"
    srt_path = out_dir / f"{stem}.srt"
    vtt_path = out_dir / f"{stem}.vtt"
    WhisperModel = load_whisper_model_class()

    print(f"[1/3] 음성 추출: {in_path.name}")
    ffmpeg_extract_audio(in_path, wav_path)

    print(f"[2/3] 음성 인식: model={args.model_size}, device={args.device}")
    model = WhisperModel(args.model_size, device=args.device, compute_type=args.compute_type)
    segments_gen, info = model.transcribe(
        str(wav_path),
        language=args.language,
        task=args.task,
        vad_filter=True,
    )
    segments = list(segments_gen)

    print(
        f"  감지 언어={info.language}, 확률={info.language_probability:.2f}, 세그먼트={len(segments)}"
    )

    print("[3/3] 결과 저장")
    write_txt(segments, txt_path)
    write_timestamped_txt(segments, timed_txt_path)
    write_srt(segments, srt_path)
    write_vtt(segments, vtt_path)

    if not args.keep_audio and wav_path.exists():
        wav_path.unlink()

    print(f"- TXT: {txt_path}")
    print(f"- Timed TXT: {timed_txt_path}")
    print(f"- SRT: {srt_path}")
    print(f"- VTT: {vtt_path}")
    if args.keep_audio:
        print(f"- WAV: {wav_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
