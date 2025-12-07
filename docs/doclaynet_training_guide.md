# DocLayNet 파인튜닝 가이드

YOLO11을 DocLayNet 데이터셋으로 파인튜닝하는 전체 과정을 설명합니다.

---

## 📋 목차

1. [사전 준비](#사전-준비)
2. [데이터 다운로드](#1-데이터-다운로드)
3. [데이터 변환](#2-데이터-변환)
4. [모델 학습](#3-모델-학습)
5. [모델 평가](#4-모델-평가)
6. [문제 해결](#문제-해결)

---

## 사전 준비

### 필수 패키지 설치

```bash
# Hugging Face datasets (DocLayNet 다운로드용)
uv pip install datasets

# PyYAML (설정 파일 읽기용)
uv pip install pyyaml

# 시각화 (평가 차트용)
uv pip install matplotlib seaborn
```

### 디렉토리 구조 확인

```
yolo/
├── data/doclaynet/          # 데이터셋 저장 위치
├── models/
│   ├── pretrained/          # 사전학습 모델
│   └── finetuned/           # 파인튜닝 결과
├── src/
│   ├── data_preparation/    # 데이터 준비 스크립트
│   └── training/            # 학습 스크립트
└── configs/                 # 설정 파일
```

---

## 1. 데이터 다운로드

### 1-1. DocLayNet 다운로드

```bash
python src/data_preparation/download_doclaynet.py --output data/doclaynet
```

**소요 시간**: 2-4시간 (인터넷 속도에 따라)

**데이터 크기**: 약 50GB

### 1-2. 다운로드 결과 확인

```bash
ls -lh data/doclaynet/
# 예상 출력:
# train/
# validation/
# test/
```

각 디렉토리 내부:
```
train/
├── images/              # PNG 이미지 파일
└── annotations.json     # COCO 형식 어노테이션
```

---

## 2. 데이터 변환

### 2-1. COCO → YOLO 형식 변환

```bash
python src/data_preparation/convert_to_yolo.py \
    --input data/doclaynet \
    --output data/doclaynet/yolo
```

**소요 시간**: 30분~1시간

### 2-2. 변환 결과 확인

```bash
ls -lh data/doclaynet/yolo/
# 예상 출력:
# images/
#   ├── train/
#   ├── val/
#   └── test/
# labels/
#   ├── train/
#   ├── val/
#   └── test/
# doclaynet.yaml
```

**YOLO 라벨 형식 예시** (`labels/train/000001.txt`):
```
8 0.5234 0.3456 0.2341 0.1234
9 0.1234 0.5678 0.4567 0.2345
```
각 줄: `<class_id> <x_center> <y_center> <width> <height>` (정규화)

### 2-3. doclaynet.yaml 확인

```bash
cat data/doclaynet/yolo/doclaynet.yaml
```

내용:
```yaml
path: data/doclaynet/yolo
train: images/train
val: images/val
test: images/test

names:
  0: Caption
  1: Footnote
  ...
  8: Table
  ...

nc: 11
```

---

## 3. 모델 학습

### 3-1. 기본 학습 (권장)

```bash
python src/training/train_doclaynet.py
```

기본 설정 (`configs/train_config.yaml` 사용):
- 모델: YOLO11n (Nano)
- Epochs: 100
- 이미지 크기: 1024
- 디바이스: MPS (Apple Silicon)
- AMP: False (MPS 미지원)

**예상 소요 시간**:
- Apple M1/M2: 20-30시간
- Apple M3: 15-20시간

### 3-2. 커스텀 설정 학습

```bash
# 에폭 수 변경
python src/training/train_doclaynet.py --epochs 50

# 배치 크기 변경
python src/training/train_doclaynet.py --batch 8

# 모델 크기 변경
python src/training/train_doclaynet.py --model yolo11s.pt

# 여러 파라미터 조합
python src/training/train_doclaynet.py \
    --epochs 50 \
    --batch 4 \
    --imgsz 1024 \
    --name my_experiment
```

### 3-3. 학습 모니터링

학습 중 출력 예시:
```
Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
  1/100     2.45G      1.234      0.567      0.890        640       1024
  2/100     2.45G      1.123      0.543      0.876        640       1024
  ...
```

**주요 지표**:
- `box_loss`: 바운딩 박스 위치 손실 (낮을수록 좋음)
- `cls_loss`: 클래스 분류 손실 (낮을수록 좋음)
- `mAP@0.5`: 검증 성능 (높을수록 좋음)

### 3-4. 학습 결과 확인

```bash
ls -lh models/finetuned/doclaynet_yolo11n/
# 예상 출력:
# weights/
#   ├── best.pt           # 최고 성능 모델
#   └── last.pt           # 마지막 에폭 모델
# results.png             # 학습 곡선 그래프
# confusion_matrix.png    # 혼동 행렬
```

---

## 4. 모델 평가

### 4-1. 검증 데이터셋 평가

```bash
python src/training/evaluate.py \
    --model models/finetuned/doclaynet_yolo11n/weights/best.pt \
    --data data/doclaynet/yolo/doclaynet.yaml \
    --split val
```

### 4-2. 테스트 데이터셋 평가

```bash
python src/training/evaluate.py \
    --model models/finetuned/doclaynet_yolo11n/weights/best.pt \
    --data data/doclaynet/yolo/doclaynet.yaml \
    --split test
```

### 4-3. 평가 결과 예시

```
📊 평가 결과
======================================================================

전체 성능:
  mAP@0.5:      0.8734
  mAP@0.5:0.95: 0.6521
  Precision:    0.8456
  Recall:       0.7823

클래스별 성능 (mAP@0.5):
  Caption        : 0.8234
  Footnote       : 0.7654
  Formula        : 0.8123
  List-item      : 0.8345
  Page-footer    : 0.8912
  Page-header    : 0.9012
  Picture        : 0.8567
  Section-header : 0.8789
  Table          : 0.9234  ⭐
  Text           : 0.8901
  Title          : 0.9123

⭐ Table 클래스 성능: 0.9234
   ✅ 우수한 표 감지 성능!
======================================================================
```

### 4-4. 여러 모델 비교

```bash
python src/training/evaluate.py \
    --model models/finetuned/yolo11n/best.pt \
    --data data/doclaynet/yolo/doclaynet.yaml \
    --compare models/finetuned/yolo11s/best.pt models/finetuned/yolo11m/best.pt
```

---

## 5. 실제 시방서 적용

### 5-1. 파인튜닝 모델로 추론

기존 `src/detection.py` 수정:

```python
from ultralytics import YOLO

class LayoutDetector:
    def __init__(self, model_path="models/finetuned/doclaynet_yolo11n/weights/best.pt"):
        """DocLayNet 파인튜닝 모델 로드"""
        self.model = YOLO(model_path)

    def detect_tables(self, image_path: str):
        """표 영역 감지"""
        results = self.model(image_path)

        tables = []
        for r in results:
            for box in r.boxes:
                # class_id == 8 (Table)
                if int(box.cls) == 8:
                    tables.append({
                        "bbox": box.xyxy[0].tolist(),
                        "confidence": float(box.conf)
                    })

        return tables
```

### 5-2. 한국어/영문 시방서 테스트

```python
from src.detection import LayoutDetector
from src.ocr import OCRProcessor
from src.parser import TableParser

# 파인튜닝 모델로 초기화
detector = LayoutDetector(model_path="models/finetuned/best.pt")
ocr = OCRProcessor()
parser = TableParser()

# 시방서 이미지 처리
image_path = "data/raw/korean_spec_page_01.png"

# 1. 표 감지
tables = detector.detect_tables(image_path)

# 2. OCR 텍스트 추출
for table in tables:
    text_data = ocr.extract_text_from_region(image_path, table['bbox'])

    # 3. 표 구조 파싱
    table_structure = parser.parse_table(text_data)

    print(table_structure)
```

---

## 문제 해결

### 문제 1: MPS 에러

**증상**:
```
RuntimeError: MPS does not support AMP
```

**해결**:
`configs/train_config.yaml`에서 `amp: false` 설정 확인

---

### 문제 2: 메모리 부족

**증상**:
```
RuntimeError: Out of memory
```

**해결**:
```bash
# 배치 크기 줄이기
python src/training/train_doclaynet.py --batch 4

# 또는 이미지 크기 줄이기
python src/training/train_doclaynet.py --imgsz 640
```

---

### 문제 3: 데이터셋 경로 에러

**증상**:
```
FileNotFoundError: data/doclaynet/yolo/doclaynet.yaml
```

**해결**:
1. COCO → YOLO 변환 완료 확인
2. YAML 파일 경로 확인
3. 절대 경로 사용:
   ```bash
   python src/training/train_doclaynet.py \
       --data /Users/.../yolo/data/doclaynet/yolo/doclaynet.yaml
   ```

---

### 문제 4: 학습이 너무 느림

**원인**: CPU 사용 중

**확인**:
```python
import torch
print(torch.backends.mps.is_available())  # True여야 함
```

**해결**:
- MPS 활성화 확인
- `--device mps` 명시적 지정
- PyTorch 2.0+ 버전 확인

---

## 다음 단계

1. ✅ DocLayNet 파인튜닝 완료
2. ⏭️ 시방서 샘플로 성능 테스트
3. ⏭️ 필요시 시방서 데이터로 추가 파인튜닝
4. ⏭️ Phase 2: 열 경계 감지 알고리즘 개선
5. ⏭️ Phase 3: 차세대 아키텍처 구현

---

## 참고 자료

- [DocLayNet 논문](https://arxiv.org/abs/2206.01062)
- [Ultralytics YOLO11 문서](https://docs.ultralytics.com/)
- [Hugging Face DocLayNet](https://huggingface.co/datasets/ds4sd/DocLayNet)
