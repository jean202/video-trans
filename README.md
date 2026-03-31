# videoTrans

MOV/MP4 같은 비디오 파일에서 음성을 추출해 텍스트와 자막 파일(SRT/VTT)을 생성하는 로컬 CLI 도구입니다.

- `transcribe.py`: 음성 -> 텍스트/자막
- `video_ocr.py`: 화면 문자열 -> 텍스트/자막(OCR)

## 1) 설치

```bash
brew install ffmpeg
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

주의:

- 새 터미널을 열 때마다 `source .venv/bin/activate`를 다시 실행해야 합니다.
- 가상환경이 비활성화된 상태에서 실행하면 `python` 또는 `faster_whisper` 관련 오류가 날 수 있습니다.
- 화면 OCR도 쓰려면 `tesseract`가 필요합니다.

```bash
brew install tesseract
```

## 2) 실행

```bash
.venv/bin/python transcribe.py input.MOV
```

가상환경을 이미 활성화했다면 아래처럼 실행해도 됩니다.

```bash
python transcribe.py input.MOV
```

기본 출력 파일:

- `input.txt`: 텍스트만
- `input.timed.txt`: `[start - end] 텍스트`
- `input.srt`: 자막(SRT)
- `input.vtt`: 자막(WebVTT)

## 3) 자주 쓰는 옵션

```bash
.venv/bin/python transcribe.py input.mp4 \
  --output-dir outputs \
  --model-size small \
  --device cpu \
  --compute-type int8 \
  --language ko \
  --task transcribe
```

- `--model-size`: `tiny`, `base`, `small`, `medium`, `large-v3`, `distil-large-v3`
- `--task translate`: 영어 번역 출력
- `--keep-audio`: 중간 WAV 파일 유지

## 4) 예시 출력

실행 후 `outputs/` 아래에 다음 파일이 생성됩니다.

- `input.txt`
- `input.timed.txt`
- `input.srt`
- `input.vtt`

## 5) 화면 문자열 OCR

비디오 안에 보이는 화면 문자열을 추출할 때는 `video_ocr.py`를 사용합니다.

```bash
python3 video_ocr.py input.MOV --output-dir outputs
```

기본 출력 파일:

- `input.ocr.txt`: 중복 병합된 OCR 텍스트
- `input.ocr.timed.txt`: 시간대별 OCR 텍스트
- `input.ocr.srt`: 시간대별 OCR 자막
- `input.ocr.cleaned.txt`: UI/노이즈를 줄인 후처리 텍스트
- `input.ocr.cleaned.timed.txt`: 후처리된 시간대별 OCR 텍스트
- `input.ocr.cleaned.srt`: 후처리된 OCR 자막
- `input.ocr.gpt_prompt.md`: ChatGPT Pro에 붙여 넣기 쉬운 정리용 프롬프트

자주 쓰는 옵션:

```bash
python3 video_ocr.py input.MOV \
  --output-dir outputs \
  --interval 1.0 \
  --lang eng \
  --psm 6 \
  --scale 2.0 \
  --keep-frames
```

- `--interval`: 몇 초마다 프레임을 샘플링할지 지정
- `--lang`: Tesseract 언어 코드. 현재 설치된 언어는 `tesseract --list-langs`로 확인
- `--min-word-confidence`: 단어 confidence가 낮은 OCR 토큰을 버려 특수문자 노이즈를 줄임
- `--char-whitelist`: 허용 문자 집합을 직접 지정하고 싶을 때 사용
- `--crop x1,y1,x2,y2`: 화면 일부만 OCR. 예를 들어 세로 숏폼 상단 문구만 읽고 싶으면 `--crop 0.05,0.05,0.95,0.35`
- `--keep-frames`: OCR에 사용한 PNG 프레임을 보존
- `--min-similarity`: OCR 노이즈가 있어도 비슷한 화면을 하나로 합칠 기준. 기본값은 `0.6`
- 기본값으로 후처리가 함께 실행되며, 원치 않으면 `--skip-postprocess`를 사용
- 기본 설치 상태가 영어(`eng`)만 포함할 수 있으므로, 한글 OCR이 필요하면 `kor` 언어 데이터를 추가로 설치해야 합니다.

노이즈가 많은 SNS/숏폼 화면이라면 아래처럼 상단 텍스트 영역만 잘라서 OCR하면 훨씬 읽기 쉬워집니다.

```bash
python3 video_ocr.py input.MOV \
  --output-dir outputs \
  --lang eng \
  --scale 3.0 \
  --min-word-confidence 55 \
  --crop 0.05,0.05,0.95,0.35
```
