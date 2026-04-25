# videoTrans

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![faster-whisper](https://img.shields.io/badge/faster--whisper-CTranslate2-orange)](https://github.com/SYSTRAN/faster-whisper)
[![Tesseract](https://img.shields.io/badge/OCR-Tesseract-4A90D9)](https://github.com/tesseract-ocr/tesseract)
[![ffmpeg](https://img.shields.io/badge/ffmpeg-audio%20extract-007808?logo=ffmpeg&logoColor=white)](https://ffmpeg.org/)

**비디오에서 자막·텍스트를 로컬에서 완전히 추출하는 CLI 도구**

외부 API 없이 로컬에서만 동작합니다. 음성 자막(Whisper)과 화면 문자열 OCR(Tesseract) 두 가지 파이프라인을 제공하며, 여러 파일을 한 번에 처리하는 배치 모드를 지원합니다.

---

## 왜 만들었나

강의·회의 녹화 영상을 정리할 때, 외부 API로 음성을 보내면 **비용이 발생하고 보안 리스크**가 있었습니다. 로컬 Whisper 모델과 Tesseract를 조합해 완전 오프라인으로 SRT/VTT 자막과 텍스트를 뽑아내는 도구가 필요해 제작했습니다.

---

## 기능 요약

| 기능 | 스크립트 | 설명 |
|---|---|---|
| 음성 → 자막/텍스트 | `transcribe.py` | ffmpeg 음성 추출 → faster-whisper 인식 → SRT/VTT/TXT 저장 |
| 화면 OCR → 텍스트 | `video_ocr.py` | 프레임 샘플링 → Tesseract OCR → 중복 병합 → UI 노이즈 후처리 |
| 배치 처리 | 두 스크립트 모두 | 여러 파일 또는 glob 패턴 한 번에 처리 |
| 번역 모드 | `transcribe.py` | `--task translate`로 외국어 음성을 영어 자막으로 변환 |

---

## 아키텍처

### 음성 자막 파이프라인 (`transcribe.py`)

```
비디오 파일 (MOV/MP4 등)
    │
    ▼  ffmpeg (-vn, 16kHz mono WAV)
오디오 추출 (WAV)
    │
    ▼  faster-whisper (CTranslate2 최적화, VAD 필터)
세그먼트 인식
    │
    ▼
┌─────────┬──────────────┬───────┬────────┐
│  .txt   │  .timed.txt  │  .srt │  .vtt  │
└─────────┴──────────────┴───────┴────────┘
```

### 화면 OCR 파이프라인 (`video_ocr.py`)

```
비디오 파일
    │
    ▼  ffmpeg (N초 간격 프레임 추출)
PNG 프레임들
    │
    ▼  Tesseract OCR (psm, 언어, scale, crop 옵션)
원시 OCR 라인
    │
    ▼  유사도 기반 중복 병합 (SequenceMatcher)
    │
    ▼  UI 노이즈 후처리 (좋아요·팔로우·시간 토큰 등 제거)
    │
    ▼
┌──────────┬──────────────────┬───────────┬──────────────────────┐
│  .ocr.txt│  .ocr.timed.txt  │  .ocr.srt │  .ocr.gpt_prompt.md  │
└──────────┴──────────────────┴───────────┴──────────────────────┘
```

---

## 기술 스택

| 구분 | 기술 |
|---|---|
| 언어 | Python 3.10+ |
| 음성 인식 | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2 최적화 Whisper) |
| OCR | Tesseract 5.x |
| 오디오 처리 | ffmpeg |
| 유사도 병합 | `difflib.SequenceMatcher` |
| 자막 포맷 | SRT, WebVTT |

---

## 출력 파일 예시

### 음성 자막 (`transcribe.py`)

**`lecture.timed.txt`**
```
[0.00 - 3.20] 안녕하세요, 오늘은 파이썬 비동기 프로그래밍에 대해 알아보겠습니다.
[3.20 - 7.80] asyncio를 사용하면 I/O 바운드 작업을 효율적으로 처리할 수 있습니다.
```

**`lecture.srt`**
```
1
00:00:00,000 --> 00:00:03,200
안녕하세요, 오늘은 파이썬 비동기 프로그래밍에 대해 알아보겠습니다.

2
00:00:03,200 --> 00:00:07,800
asyncio를 사용하면 I/O 바운드 작업을 효율적으로 처리할 수 있습니다.
```

### 화면 OCR (`video_ocr.py`)

**`shorts.ocr.cleaned.txt`**
```
Wait, you can do this in Python?
No way this actually works
Let me show you the trick
```

---

## Whisper 모델 비교

| 모델 | 크기 | CPU 속도\* | 정확도 | 추천 용도 |
|---|---|---|---|---|
| `tiny` | ~75 MB | 매우 빠름 | 낮음 | 빠른 초안 |
| `base` | ~145 MB | 빠름 | 보통 | 간단한 메모 |
| `small` | ~465 MB | 보통 | 좋음 | **일반 사용 권장** |
| `medium` | ~1.5 GB | 느림 | 높음 | 중요 강의·회의 |
| `large-v3` | ~3 GB | 매우 느림 | 최고 | 고정밀 필요 시 |
| `distil-large-v3` | ~1.5 GB | 빠름 | large급 | GPU 환경 권장 |

\* CPU int8 연산 기준 실시간 배율

---

## 설치

```bash
brew install ffmpeg
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

화면 OCR까지 사용하려면:

```bash
brew install tesseract
brew install tesseract-lang  # 한글 등 추가 언어
```

> 새 터미널을 열 때마다 `source .venv/bin/activate`를 다시 실행해야 합니다.

---

## 사용법

### 음성 자막 (`transcribe.py`)

```bash
# 단일 파일
.venv/bin/python transcribe.py input.MOV --output-dir outputs

# 여러 파일 배치
.venv/bin/python transcribe.py *.mp4 --output-dir outputs

# 번역 모드 (외국어 → 영어 자막)
.venv/bin/python transcribe.py input.mp4 --task translate --language en

# 고정밀 모드
.venv/bin/python transcribe.py input.mp4 \
  --model-size medium \
  --language ko \
  --output-dir outputs
```

**옵션 목록**

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--model-size` | `small` | `tiny` / `base` / `small` / `medium` / `large-v3` / `distil-large-v3` |
| `--language` | 자동 감지 | 언어 코드 (예: `ko`, `en`) |
| `--task` | `transcribe` | `transcribe` (원문) 또는 `translate` (영어 번역) |
| `--device` | `cpu` | `cpu` / `cuda` / `auto` |
| `--compute-type` | `int8` | `int8` / `float16` / `float32` |
| `--keep-audio` | false | 중간 WAV 파일 보존 |
| `--output-dir` | `.` | 결과 저장 경로 |

---

### 화면 OCR (`video_ocr.py`)

```bash
# 기본 실행
python3 video_ocr.py input.MOV --output-dir outputs

# 숏폼 SNS 영상 — 상단 자막 영역만 OCR
python3 video_ocr.py shorts.mp4 \
  --output-dir outputs \
  --lang eng \
  --scale 3.0 \
  --min-word-confidence 55 \
  --crop 0.05,0.05,0.95,0.35
```

**옵션 목록**

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--interval` | `1.0` | 프레임 샘플링 간격 (초) |
| `--lang` | `eng` | Tesseract 언어 코드 (한글: `kor`) |
| `--scale` | `1.0` | 이미지 확대 배율 (OCR 정확도 향상) |
| `--crop` | 전체 | 관심 영역 `x1,y1,x2,y2` (0~1 비율) |
| `--min-word-confidence` | `0` | 낮은 confidence 토큰 제거 임계값 |
| `--min-similarity` | `0.6` | 중복 병합 유사도 기준 |
| `--psm` | `6` | Tesseract PSM 모드 |
| `--keep-frames` | false | 추출된 PNG 프레임 보존 |
| `--skip-postprocess` | false | UI 노이즈 후처리 건너뜀 |

---

## 출력 파일 목록

### `transcribe.py`

| 파일 | 설명 |
|---|---|
| `{stem}.txt` | 순수 텍스트 |
| `{stem}.timed.txt` | `[start - end] 텍스트` 형식 |
| `{stem}.srt` | SRT 자막 |
| `{stem}.vtt` | WebVTT 자막 |

### `video_ocr.py`

| 파일 | 설명 |
|---|---|
| `{stem}.ocr.txt` | 중복 병합된 OCR 텍스트 |
| `{stem}.ocr.timed.txt` | 시간대별 OCR 텍스트 |
| `{stem}.ocr.srt` | OCR 자막 |
| `{stem}.ocr.cleaned.txt` | UI 노이즈 제거 후처리 텍스트 |
| `{stem}.ocr.cleaned.timed.txt` | 후처리 시간대별 텍스트 |
| `{stem}.ocr.cleaned.srt` | 후처리 SRT 자막 |
| `{stem}.ocr.gpt_prompt.md` | ChatGPT에 바로 붙여넣기 가능한 정리 프롬프트 |
