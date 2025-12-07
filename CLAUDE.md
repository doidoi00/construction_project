# 건설 시방서 표 구조화 프로젝트

## 프로젝트 목표
YOLO와 OCR을 활용하여 문서의 레이아웃을 분류하고, 건설 시방서의 표를 인식하여 구조화된 JSON 파일로 추출하는 시스템 개발

## 개발 환경

### 학습 및 추론 환경
- **플랫폼**: Google Colab Pro/Pro+
- **GPU**: NVIDIA A100 (40GB/80GB VRAM)
- **CUDA**: 11.x/12.x 지원
- **Python 버전**: 3.10+
- **패키지 관리**: pip (Colab 기본)
- **장점**:
  - 고성능 GPU 무료/저렴하게 사용
  - CUDA 완전 지원
  - 대용량 데이터셋 학습 가능
  - 배치 크기 확대 가능
  - 학습과 추론을 동일 환경에서 수행

## 기술 스택
- **언어**: Python
- **핵심 기술**:
  - YOLO11: 문서 레이아웃 분류 및 표/도식 영역 감지
  - Table Transformer (Microsoft): 표 내부 구조 인식 (행/열/셀/병합 셀)
  - OCR (EasyOCR): 텍스트 추출
  - OpenCV: 이미지 처리
  - JSON: 데이터 구조화
- **학습 데이터**:
  - DocLayNet (YOLO 파인튜닝)
  - PubTables-1M (Table Transformer 사전학습)
- **커스텀 학습**: 시방서 특화 레이아웃 분류 추가 학습 가능

## 프로젝트 구조
```
yolo/
├── data/
│   ├── raw/              # 원본 시방서 (한국어/영문)
│   ├── doclaynet/        # DocLayNet 데이터셋 (YOLO 형식)
│   └── output/           # 최종 JSON 결과
├── models/
│   ├── pretrained/       # 사전학습 모델 (yolo11n.pt)
│   ├── finetuned/        # 파인튜닝 결과 (best.pt, last.pt)
│   └── final/            # 최종 배포 모델
├── src/
│   ├── detection.py          # YOLO 레이아웃/표 감지
│   ├── table_structure.py    # Table Transformer 구조 인식
│   ├── ocr.py                # OCR 텍스트 추출
│   ├── parser.py             # 데이터 구조화 및 JSON 변환
│   ├── training.py           # 커스텀 모델 학습
│   └── utils.py              # 유틸리티 함수
├── configs/              # 학습/추론 설정
├── tests/                # 테스트 파일
├── docs/                 # 문서화
├── main.py               # 메인 실행 파일
└── pyproject.toml        # 의존성 관리
```

## 핵심 파이프라인 (현재 구현)

### 고급 파이프라인 (Table Transformer 통합) ✨ NEW
1. **레이아웃 분류**: YOLO11로 문서 레이아웃 분석 (DocLayNet 파인튜닝)
2. **표 영역 감지**: YOLO로 표 영역 정밀 탐지
3. **표 구조 인식**: Table Transformer로 표 내부 구조 분석 (행/열/셀/병합 셀)
4. **텍스트 추출**: OCR로 감지된 표 영역에서 텍스트 추출
5. **셀 매핑**: OCR 텍스트를 Table Transformer가 인식한 셀 그리드에 매핑
6. **JSON 변환**: 구조화된 표 데이터를 JSON 형식으로 저장

### 기본 파이프라인 (레거시, --no-structure 옵션)
1. **레이아웃 분류**: YOLO로 문서의 전체 레이아웃 분석
2. **표 영역 감지**: 분류된 레이아웃 중 표 영역 정밀 탐지
3. **텍스트 추출**: OCR로 감지된 표 영역에서 텍스트 추출
4. **구조 분석**: 텍스트의 위치(x, y 좌표) 기반으로 행/열 구조 파악
5. **JSON 변환**: 파싱된 데이터를 구조화된 JSON 형식으로 저장

### 현재 구현 상태 (2025-12-07)

#### ✅ 완료된 기능
1. **YOLO 문서 레이아웃 감지**
   - DocLayNet 데이터셋으로 파인튜닝 완료
   - 모델: `models/finetuned/weights/best.pt`
   - 11개 레이아웃 클래스 인식 (Table, Text, Title 등)

2. **Table Transformer 구조 인식**
   - Microsoft 사전학습 모델 통합 완료
   - 행/열/셀 자동 감지
   - 병합 셀 자동 인식
   - 모듈: `src/table_structure.py`

3. **OCR 텍스트 추출**
   - EasyOCR 한국어/영문 동시 지원
   - 텍스트 → 셀 그리드 자동 매핑

4. **통합 파이프라인**
   - YOLO → Table Transformer → OCR → JSON 출력
   - 병합 셀 포함 구조화된 JSON 생성
   - CLI 인터페이스 제공 (`main.py`)

#### 🔄 알고리즘 선택: Table Transformer vs TableCenterNet

| 비교 항목 | Table Transformer (현재) | TableCenterNet |
|----------|------------------------|----------------|
| 아키텍처 | DETR 기반 2-stage | CenterNet 기반 1-stage |
| 학습 난이도 | 중간 | 쉬움 (1-stage) |
| 추론 속도 | 중간 | 빠름 |
| 정확도 | 높음 (PubTables-1M) | SOTA (TableGraph-24k) |
| 복잡한 표 | 강건함 | 매우 강건함 |
| 사전학습 모델 | Hugging Face 제공 | 직접 구현 필요 |
| 통합 용이성 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

**현재 선택: Table Transformer**
- Hugging Face 사전학습 모델 사용 가능 (즉시 적용)
- 문서화 및 커뮤니티 지원 우수
- PubTables-1M 데이터셋 기반으로 학술 문서 특화

**향후 고려: TableCenterNet**
- 더 빠른 추론 속도 필요 시
- 복잡한 표 처리 정확도 향상 필요 시
- A100 GPU로 직접 학습 가능 시

#### ⚠️ 현재 제약사항

**기술적 제약:**
- **Table Transformer 한계**: 매우 복잡한 중첩 구조나 비정형 표에서 정확도 저하 가능
- **OCR 매핑 오류**: 셀 경계와 텍스트 위치가 불일치할 경우 잘못된 셀에 매핑
- **이미지 처리**: 표 내부 이미지는 빈 셀로 처리 (텍스트만 추출)

**데이터 제약:**
- 건설 시방서 특화 학습 데이터 부족
- DocLayNet은 학술/법률 문서 위주 (건설 문서 비율 낮음)

---

## 개선 로드맵

### Phase 1: DocLayNet 파인튜닝 ✅ 완료

**목표**: YOLO11을 DocLayNet 데이터셋으로 파인튜닝하여 문서 레이아웃 분류 정확도 향상

**상태**: Google Colab A100 GPU로 학습 완료 (2025-12-07)

#### DocLayNet 데이터셋 개요

- **규모**: 80,863 페이지 (학술논문, 재무보고서, 매뉴얼, 법률문서 등)
- **클래스**: 11개 레이아웃 요소
  - Caption, Footnote, Formula, List-item, Page-footer, Page-header, Picture, Section-header, Table, Text, Title
- **형식**: COCO JSON (바운딩 박스 + 클래스 라벨)
- **특징**: 다양한 문서 타입, 복잡한 레이아웃, 실제 문서 기반

#### 구현 계획

**1. 데이터 준비 (1-2일)**

```bash
# DocLayNet 다운로드
python src/data_preparation/download_doclaynet.py

# COCO → YOLO 형식 변환
python src/data_preparation/convert_to_yolo.py \
  --input data/doclaynet/raw \
  --output data/doclaynet/yolo
```

**주요 작업**:
- COCO JSON → YOLO txt 형식 변환 (클래스 ID, 정규화된 bbox)
- train/val/test 분할 (기본 제공 분할 사용)
- `doclaynet.yaml` 설정 파일 생성 (경로, 클래스 정의)

**2. 모델 학습** ✅ 완료

**학습 환경**: Google Colab A100 GPU

**학습 설정**:
```yaml
model: models/pretrained/yolo11n.pt  # Nano 모델
data: data/doclaynet/doclaynet.yaml
epochs: 100
batch: 32           # A100 40GB 기준
imgsz: 1024         # 문서 이미지 크기
device: 0           # CUDA GPU 0
amp: true           # Mixed Precision (A100 지원)
workers: 4
patience: 20        # Early stopping
save_period: 10     # 체크포인트 저장 주기
```

**학습 명령어** (Colab):
```bash
# GPU 확인
!nvidia-smi

# YOLO11 DocLayNet 파인튜닝
!python src/training/train_doclaynet.py \
  --device 0 \
  --batch 32 \
  --epochs 100
```

**3. 평가 및 검증 (1일)**

```bash
# 모델 평가
python src/training/evaluate.py \
  --model models/finetuned/best.pt \
  --data data/doclaynet/doclaynet.yaml

# 시각화
python src/training/visualize.py \
  --model models/finetuned/best.pt \
  --images data/doclaynet/yolo/images/test
```

**평가 지표**:
- mAP@0.5, mAP@0.5:0.95
- 클래스별 Precision, Recall, F1
- Table 클래스 특화 분석

**4. 시방서 적용 (2-3일)**

- 한국어/영문 건설 시방서 샘플로 테스트
- 필요시 시방서 특화 데이터 추가 학습
- 기존 파이프라인 통합 (`src/detection.py` 업데이트)

#### 예상 성과

- **레이아웃 분류 정확도**: 85% → 95% 향상 (추정)
- **표 영역 감지**: 더 정밀한 바운딩 박스
- **복잡한 문서 처리**: 다양한 레이아웃 패턴 대응
- **일반화 성능**: 다양한 문서 타입에 강건

#### 기술적 고려사항

**Google Colab GPU 설정**:
```python
import torch

# GPU 사용 가능 확인
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    device = torch.device("cpu")

# YOLO 학습 시 (A100 최적화)
model.train(
    data="data/doclaynet/doclaynet.yaml",
    device=0,           # CUDA GPU 0
    batch=32,           # A100 40GB 기준 최적 배치 크기
    amp=True,           # Mixed Precision Training
    cache=True          # 데이터 캐싱으로 속도 향상
)
```

**데이터 형식 변환**:
```python
# COCO bbox → YOLO bbox
# COCO: [x_min, y_min, width, height] (픽셀 단위)
# YOLO: [x_center, y_center, width, height] (정규화)

x_center = (x_min + width/2) / img_width
y_center = (y_min + height/2) / img_height
norm_width = width / img_width
norm_height = height / img_height
```

### Phase 1.5: Table Transformer 통합 ✅ 완료

**목표**: 표 구조 인식 정확도 향상

**완료 사항**:
- Microsoft Table Transformer 통합
- 행/열/셀 자동 감지
- 병합 셀 자동 인식
- OCR 텍스트 → 셀 매핑
- 모듈: `src/table_structure.py`

---

### Phase 2: 건설 시방서 특화 학습 (현재 우선순위)

**목표**: 건설 시방서 문서에 특화된 모델 성능 향상

#### 2.1 건설 시방서 데이터셋 구축

**데이터 수집**:
- 한국어 건설 시방서: 국가건설기준센터, 공공기관 시방서
- 영문 건설 시방서: 호주, 캐나다, 미국 등 (현재 보유)
- 목표: 1,000+ 페이지 (다양한 표 유형 포함)

**라벨링 작업**:
1. **문서 레이아웃**: YOLO 형식
   - 기존 DocLayNet 모델로 자동 라벨링
   - 수동 검수 및 보정

2. **표 구조**: Table Transformer 형식 (선택)
   - TableCenterNet 학습 시 필요
   - 행/열/셀/병합 셀 라벨링

**도구**:
- CVAT 또는 Label Studio 사용
- 자동 라벨링 + 수동 검수 조합

#### 2.2 YOLO 추가 파인튜닝

```bash
# Google Colab에서 실행
!python src/training/train_construction.py \
  --base-model models/finetuned/weights/best.pt \
  --data data/construction/construction.yaml \
  --epochs 50 \
  --device 0
```

**기대 효과**:
- 건설 시방서 특유의 표 레이아웃 인식 향상
- False Positive 감소

#### 2.3 TableCenterNet 실험 (선택)

**목표**: Table Transformer 대비 성능 비교

**구현 계획**:
1. TableCenterNet 논문 구현체 확보
2. 건설 시방서 데이터로 학습 (A100 GPU)
3. Table Transformer와 정량적 비교
   - 정확도 (mAP, GriTS 등)
   - 추론 속도
   - 복잡한 표 처리 능력

**선택 기준**:
- TableCenterNet이 유의미하게 우수하면 전환
- 그렇지 않으면 Table Transformer 유지

---

### Phase 3: 고급 기능 개선 (중기)

#### 3.1 표 내부 이미지 처리

**목표**: 표 내 도면/이미지 추출 및 설명 생성

**구현**:
1. YOLO에 `table-image` 클래스 추가
2. GPT-4V 또는 Gemini Vision API 통합
3. JSON에 이미지 설명 필드 추가

**예시 출력**:
```json
{
  "cell_value": "",
  "cell_type": "image",
  "image": {
    "bbox": [100, 200, 300, 400],
    "description": "철근 배근 상세도",
    "extracted_path": "images/table1_cell_2_3.png"
  }
}
```

#### 3.2 OCR 후처리 개선

1. **텍스트 분리**: "철근D13" → ["철근", "D13"]
2. **단위 정규화**: "1,000mm" → {"value": 1000, "unit": "mm"}
3. **수식 인식**: LaTeX 변환 (Mathpix 또는 유사 도구)

#### 3.3 PDF 직접 처리

**목표**: 이미지 변환 없이 PDF → JSON

**도구**:
- pdfplumber 또는 PyMuPDF
- 벡터 정보 활용으로 정확도 향상

---

### Phase 4: 프로덕션 배포 (장기)

#### 4.1 API 서버 구축

**프레임워크**: FastAPI

**엔드포인트**:
```
POST /api/v1/extract
  - Input: PDF/이미지 파일
  - Output: 구조화된 JSON

GET /api/v1/status/{job_id}
  - 비동기 처리 상태 확인
```

#### 4.2 웹 인터페이스

**기능**:
- 파일 업로드
- 실시간 처리 진행률
- JSON 다운로드
- 표 시각화 (미리보기)

#### 4.3 배치 처리 시스템

**도구**: Celery + Redis

**기능**:
- 대량 문서 일괄 처리
- 우선순위 큐
- 오류 재처리

## AI 지원 시 필수 지침

### 1. 최신 정보 활용
- 코드 작성 전 관련 라이브러리의 최신 문서를 웹에서 검색하여 확인할 것
- 특히 YOLO, PyTorch (MPS 지원), EasyOCR의 최신 버전 및 macOS 호환성 확인
- DocLayNet 데이터셋의 최신 사용법 및 형식 검색

### 2. Skill 활용
- 코드 작성 시 관련 Skill이 있다면 적극 활용
- 특히 Python, 이미지 처리, 머신러닝 관련 Skill 우선 참고

### 3. GPU 환경 최적화 (Google Colab)

- **GPU**: NVIDIA A100 (CUDA 11.x/12.x)
- Mixed Precision Training (AMP) 활성화
- 배치 크기 최대화 (메모리 허용 범위 내)
- 코드 예시:
```python
import torch

# GPU 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 학습 및 추론
model.train(data=config, device=0, amp=True, batch=32)
```

### 4. 패키지 관리
- **Colab 환경**: pip 사용 (`!pip install <package>`)
- 필요 패키지:
  - `ultralytics` (YOLO11)
  - `transformers` (Table Transformer)
  - `timm` (PyTorch Image Models)
  - `easyocr` (OCR)
  - `opencv-python` (이미지 처리)

### 5. 코드 작성 원칙
- Python 기초 문법은 이해하고 있으나, 심화 기능(클래스 설계, 고급 라이브러리 사용 등)에는 **상세한 설명** 필요
- **각 코드 블록마다 주석으로 다음 내용 포함**:
  - 이 코드가 무엇을 하는지
  - 왜 이렇게 구현했는지
  - 주요 변수/함수의 역할
  - 입력과 출력 형식
- 복잡한 알고리즘은 단계별로 설명
- 에러 처리 및 예외 상황 고려

### 6. DocLayNet 데이터셋 활용
- DocLayNet의 레이아웃 분류 체계 이해 및 활용
- 필요 시 시방서 특화 클래스 추가 학습 방법 제시
- 데이터 전처리 및 증강 기법 적용

## 현재 중점 과제

### ✅ 완료된 과제 (Phase 1 & 1.5)

1. **DocLayNet 파인튜닝**
   - ✅ Google Colab A100 GPU로 학습 완료
   - ✅ 모델: `models/finetuned/weights/best.pt`
   - ✅ 11개 레이아웃 클래스 인식

2. **Table Transformer 통합**
   - ✅ Microsoft 사전학습 모델 통합
   - ✅ 행/열/셀 자동 감지
   - ✅ 병합 셀 자동 인식
   - ✅ OCR 텍스트 → 셀 매핑
   - ✅ 모듈: `src/table_structure.py`

3. **통합 파이프라인**
   - ✅ YOLO → Table Transformer → OCR → JSON
   - ✅ CLI 인터페이스 (`main.py`)
   - ✅ Google Colab 실행 환경 구축

---

### 🔥 최우선 과제 (Phase 2 - 현재)

#### 1. 건설 시방서 데이터셋 구축 및 특화 학습

**목표**: 건설 시방서에 특화된 모델 성능 향상

**세부 과제**:
- [ ] **데이터 수집** (목표: 1,000+ 페이지)
  - [ ] 한국어 건설 시방서 PDF 수집
  - [ ] 기존 영문 시방서 정리 (data/raw/pdf/)
  - [ ] 표가 많이 포함된 페이지 우선 선별

- [ ] **데이터 라벨링**
  - [ ] CVAT 또는 Label Studio 설정
  - [ ] DocLayNet 모델로 자동 라벨링
  - [ ] 수동 검수 및 보정 (표 영역 중심)

- [ ] **YOLO 추가 파인튜닝**
  - [ ] 건설 시방서 데이터셋 준비
  - [ ] Google Colab A100으로 학습
  - [ ] 성능 평가 (mAP, Precision, Recall)

- [ ] **TableCenterNet 실험** (선택)
  - [ ] 구현체 확보 및 설정
  - [ ] 건설 시방서 데이터로 학습
  - [ ] Table Transformer와 비교

**우선순위**: ⭐⭐⭐⭐⭐

---

#### 2. 실제 건설 시방서 테스트 및 검증

**목표**: 실제 문서로 파이프라인 검증 및 개선

**세부 과제**:
- [ ] **PDF → 이미지 변환**
  - [ ] PDF 페이지 추출 스크립트 개선
  - [ ] 고해상도 이미지 생성 (DPI 300+)

- [ ] **전체 파이프라인 실행**
  - [ ] 한국어 시방서 10+ 문서 처리
  - [ ] 영문 시방서 10+ 문서 처리
  - [ ] JSON 출력 품질 평가

- [ ] **오류 분석 및 개선**
  - [ ] False Positive/Negative 분석
  - [ ] 병합 셀 인식 오류 케이스 수집
  - [ ] OCR 매핑 오류 원인 분석

**우선순위**: ⭐⭐⭐⭐⭐

---

### 📋 중기 과제 (Phase 3)

#### 3. 고급 기능 구현

- [ ] **표 내부 이미지 처리**
  - [ ] YOLO 클래스 확장 (table-image)
  - [ ] Vision API 통합 (GPT-4V/Gemini)
  - [ ] JSON 형식 확장

- [ ] **OCR 후처리**
  - [ ] 텍스트 분리 알고리즘
  - [ ] 단위 정규화
  - [ ] 수식 인식

- [ ] **PDF 직접 처리**
  - [ ] pdfplumber/PyMuPDF 통합
  - [ ] 벡터 정보 활용

**우선순위**: ⭐⭐⭐

---

### 🚀 장기 과제 (Phase 4)

#### 4. 프로덕션 배포

- [ ] FastAPI 서버 구축
- [ ] 웹 인터페이스 개발
- [ ] 배치 처리 시스템
- [ ] Docker 컨테이너화

**우선순위**: ⭐⭐

## 예상 출력 JSON 형식 예시
```json
{
  "document_id": "spec_001",
  "layout": [
    {
      "type": "table",
      "bbox": [x1, y1, x2, y2],
      "content": {
        "rows": 5,
        "cols": 3,
        "data": [
          ["헤더1", "헤더2", "헤더3"],
          ["데이터1", "데이터2", "데이터3"]
        ],
        "merged_cells": []
      }
    }
  ]
}
```

---

## 참고 자료

### 논문 및 기술 문서

1. **Table Transformer**
   - [PubTables-1M 논문](https://arxiv.org/abs/2110.00061)
   - [Hugging Face 모델 카드](https://huggingface.co/docs/transformers/main/en/model_doc/table-transformer)
   - [Microsoft GitHub](https://github.com/microsoft/table-transformer)

2. **TableCenterNet**
   - [TableCenterNet 논문](https://arxiv.org/abs/2504.17522)
   - One-stage 네트워크로 더 빠른 추론 속도
   - TableGraph-24k 데이터셋 SOTA

3. **DocLayNet**
   - [DocLayNet 데이터셋](https://github.com/DS4SD/DocLayNet)
   - 80,863 페이지, 11개 레이아웃 클래스
   - COCO JSON 형식

4. **YOLO**
   - [Ultralytics YOLO11](https://docs.ultralytics.com/)
   - Document Layout Analysis 가이드

### 개발 환경 설정

1. **Google Colab**
   - [Colab GPU 타입](https://colab.research.google.com/signup)
   - [PyTorch CUDA 설정](https://pytorch.org/get-started/locally/)

### 프로젝트 문서

- [Table Transformer 사용 가이드](docs/table_transformer_usage.md)
- [DocLayNet 학습 가이드](docs/doclaynet_training_guide.md)
- [Colab 학습 가이드](docs/colab_training_guide.md)

---

**모든 코드 작성, 디버깅, 구조 개선 시 위 지침을 참고하고, 특히 Google Colab A100 GPU 환경 최적화와 최신 정보 검색을 필수로 수행할 것**