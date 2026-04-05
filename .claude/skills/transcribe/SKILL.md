---
name: transcribe
description: 영상에서 음성을 추출하여 텍스트/자막(SRT/VTT)으로 변환하거나, 화면 OCR을 수행한다.
argument-hint: "[영상 파일 경로]"
disable-model-invocation: true
---

## 영상 텍스트 추출

대상: **$ARGUMENTS**

### 음성→텍스트 (Speech-to-Text)
```bash
python transcribe.py --input video.mp4 --output subtitles.srt
python transcribe.py --input video.mov --format vtt    # VTT 형식
```

### 화면 OCR
```bash
python video_ocr.py --input video.mp4    # 화면 텍스트 추출
```

### 옵션
- `--model`: Whisper 모델 크기 (tiny/base/small/medium/large)
- `--format`: 출력 형식 (srt/vtt/txt)
- `--lang`: 언어 지정 (ko/en/ja 등)

### 기술 스택
- Faster-Whisper (음성 인식)
- Tesseract (OCR)
- FFmpeg (영상 처리)
