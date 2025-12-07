# 건설 시방서 표 구조화 프로젝트

YOLO, Table Transformer, OCR을 활용하여 건설 시방서의 표를 인식하고 구조화된 JSON 파일로 추출하는 시스템

## 주요 특징

- **고정밀 표 감지**: YOLO11 + DocLayNet 파인튜닝 모델
- **표 구조 인식**: Microsoft Table Transformer로 행/열/셀/병합 셀 자동 감지
- **다국어 OCR**: 한국어/영어 동시 지원
- **구조화된 출력**: 병합 셀 정보를 포함한 JSON 형식

## 핵심 파이프라인

### 고급 파이프라인 (기본값)
```
PDF/이미지
    ↓
[1] YOLO 표 영역 감지 (DocLayNet 파인튜닝)
    ↓
[2] Table Transformer 구조 인식 (행/열/셀/병합 셀)
    ↓
[3] OCR 텍스트 추출 (EasyOCR)
    ↓
[4] 텍스트 → 셀 그리드 매핑
    ↓
[5] 구조화된 JSON 출력
```

### 기존 방식 (`--no-structure` 옵션)
```
PDF/이미지
    ↓
[1] YOLO 표 영역 감지
    ↓
[2] OCR 텍스트 추출
    ↓
[3] 좌표 기반 행/열 클러스터링
    ↓
[4] JSON 출력
```

## 개발 환경

- **학습/추론**: Google Colab Pro/Pro+ (NVIDIA A100 GPU)
- **Python**: 3.10+
- **CUDA**: 11.x/12.x 지원
- **패키지 관리**: pip

## 빠른 시작

### 1. 저장소 클론
```bash
git clone <repository-url>
cd yolo
```

### 2. 패키지 설치 (Google Colab)
```python
# Colab 노트북에서 실행
!pip install ultralytics transformers timm easyocr opencv-python pillow pyyaml numpy datasets tqdm matplotlib seaborn
```

### 3. 모델 파일 준비
```bash
# YOLO 파인튜닝 모델을 models/finetuned/weights/best.pt에 업로드
# (Google Drive 또는 Colab 파일 시스템)
```

## 사용 방법

### 기본 사용법

```bash
# 단일 이미지 처리
python main.py data/test_table.png

# 디렉토리 일괄 처리
python main.py data/images/ -o data/output

# 출력 폴더 지정
python main.py data/images/ -o data/output/
```

### 고급 옵션

```bash
# YOLO 및 Table Transformer 신뢰도 조정
python main.py data/test_table.png \
  --conf 0.5 \              # YOLO 신뢰도 임계값
  --structure-conf 0.7      # Table Transformer 신뢰도 임계값

# 기존 방식 사용 (Table Transformer 비활성화)
python main.py data/test_table.png --no-structure

# OCR 언어 설정
python main.py data/test_table.png --lang ko,en

# 디바이스 지정
python main.py data/test_table.png --device cuda  # Colab GPU
python main.py data/test_table.png --device cpu   # CPU

# 커스텀 YOLO 모델 사용
python main.py data/test_table.png -m models/custom/weights/best.pt

# 도움말 확인
python main.py --help
```

## 프로젝트 구조

```
yolo/
├── data/
│   ├── raw/              # 원본 시방서 (PDF)
│   ├── doclaynet/        # DocLayNet 데이터셋 (YOLO 학습용)
│   ├── construction/     # 건설 시방서 데이터셋 (커스텀 학습용)
│   └── output/           # 최종 JSON 결과
├── models/
│   ├── pretrained/       # 사전학습 모델 (yolo11n.pt)
│   ├── finetuned/        # 파인튜닝 결과 (best.pt)
│   └── final/            # 최종 배포 모델
├── src/
│   ├── detection.py          # YOLO 레이아웃/표 감지
│   ├── table_structure.py    # Table Transformer 구조 인식
│   ├── ocr.py                # OCR 텍스트 추출
│   ├── parser.py             # 데이터 구조화 및 JSON 변환
│   ├── training.py           # 커스텀 모델 학습
│   └── utils.py              # 유틸리티 함수
├── configs/              # 학습/추론 설정
├── docs/                 # 문서화
│   ├── table_transformer_usage.md
│   ├── doclaynet_training_guide.md
│   └── colab_training_guide.md
├── notebooks/            # Colab 노트북
├── main.py               # 메인 실행 파일
├── test_table_structure.py  # 테스트 스크립트
├── CLAUDE.md             # AI 지원 가이드
└── README.md             # 프로젝트 설명
```

## 출력 JSON 형식

```json
{
  "document_id": "test_table",
  "metadata": {
    "created_at": "2025-12-07T12:00:00",
    "table_count": 1
  },
  "layout": [
    {
      "type": "table",
      "bbox": [100, 200, 800, 600],
      "confidence": 0.95,
      "content": {
        "rows": 5,
        "cols": 3,
        "data": [
          ["헤더1", "헤더2", "헤더3"],
          ["데이터1-1", "데이터1-2", "데이터1-3"],
          ["데이터2-1", "데이터2-2", "데이터2-3"]
        ],
        "merged_cells": [
          {
            "start_row": 0,
            "start_col": 1,
            "end_row": 0,
            "end_col": 2
          }
        ]
      }
    }
  ]
}
```

## 주요 기능

### 1. YOLO 기반 표 감지
- DocLayNet 데이터셋으로 파인튜닝 완료
- 11개 레이아웃 클래스 인식 (Table, Text, Title 등)
- 높은 정확도의 표 영역 탐지

### 2. Table Transformer 구조 인식
- Microsoft 사전학습 모델 사용
- 행/열 경계 자동 감지
- 병합 셀 자동 인식
- 복잡한 표 구조 강건한 처리

### 3. EasyOCR 텍스트 추출
- 한국어 + 영어 동시 지원
- 텍스트 바운딩 박스 정보 제공
- 표 영역 내 정밀 추출

### 4. 텍스트 → 셀 매핑
- OCR 텍스트를 셀 그리드에 자동 매핑
- 중심점 기반 매핑 알고리즘
- 병합 셀 처리

### 5. 구조화된 JSON 출력
- 2D 배열 형식 표 데이터
- 병합 셀 메타데이터 포함
- 메타데이터 (생성 시간, 표 개수 등)

## 테스트

### Table Transformer 기능 테스트

```bash
# 모듈 단독 테스트
python test_table_structure.py

# 전체 파이프라인 테스트 정보
python test_table_structure.py --full
```

### 실제 문서 테스트

```bash
# 건설 시방서 PDF에서 표 추출
python main.py data/raw/pdf/spec_document.png
```

## 개발 로드맵

### Phase 1: DocLayNet 파인튜닝 ✅ 완료
- [x] DocLayNet 데이터셋 다운로드 및 전처리
- [x] YOLO11 모델 파인튜닝 (Google Colab A100)
- [x] 모델 평가 및 검증
- [x] 시방서 적용 테스트

### Phase 1.5: Table Transformer 통합 ✅ 완료
- [x] Microsoft Table Transformer 통합
- [x] 행/열/셀 자동 감지
- [x] 병합 셀 인식
- [x] OCR 텍스트 → 셀 매핑
- [x] 통합 파이프라인 구축

### Phase 2: 건설 시방서 특화 학습 (진행 중)
- [ ] 건설 시방서 데이터셋 구축 (1,000+ 페이지)
- [ ] 데이터 라벨링 (CVAT/Label Studio)
- [ ] YOLO 추가 파인튜닝
- [ ] TableCenterNet 실험 (선택)
- [ ] 성능 평가 및 비교

### Phase 3: 고급 기능 개선 (중기)
- [ ] 표 내부 이미지 처리 (Vision API 통합)
- [ ] OCR 후처리 (텍스트 분리, 단위 정규화)
- [ ] PDF 직접 처리 (벡터 정보 활용)

### Phase 4: 프로덕션 배포 (장기)
- [ ] FastAPI 서버 구축
- [ ] 웹 인터페이스 개발
- [ ] 배치 처리 시스템
- [ ] Docker 컨테이너화

## 주요 기술 스택

- **YOLO11**: Ultralytics (문서 레이아웃 감지)
- **Table Transformer**: Microsoft/Hugging Face (표 구조 인식)
- **OCR**: EasyOCR (한글/영어 텍스트 추출)
- **이미지 처리**: OpenCV, Pillow
- **딥러닝**: PyTorch (CUDA 지원)
- **PDF 처리**: PyMuPDF (계획 중)

## 성능 최적화

### Google Colab A100 GPU 활용

```python
# GPU 확인
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 학습 시 Mixed Precision 활성화
model.train(
    data="data/doclaynet/doclaynet.yaml",
    device=0,
    batch=32,
    amp=True,
    cache=True
)
```

### 신뢰도 임계값 조정

- **높은 정확도 필요**: `--conf 0.7 --structure-conf 0.8`
- **높은 재현율 필요**: `--conf 0.3 --structure-conf 0.5`
- **권장값**: `--conf 0.5 --structure-conf 0.7`

## 문제 해결

### Q1: 표가 감지되지 않아요
```bash
# YOLO 신뢰도 임계값 낮추기
python main.py input.png --conf 0.3
```

### Q2: 행/열이 제대로 감지되지 않아요
```bash
# Table Transformer 신뢰도 임계값 조정
python main.py input.png --structure-conf 0.5
```

### Q3: 병합 셀이 인식되지 않아요
- 이미지 해상도를 높여보세요 (권장: 300 DPI 이상)
- 신뢰도 임계값을 낮춰보세요

### Q4: OCR 텍스트가 잘못된 셀에 매핑돼요
- Table Transformer 구조 인식 결과를 먼저 확인하세요
- OCR 바운딩 박스 정확도를 확인하세요
- 이미지 품질을 개선하세요

### Q5: GPU 메모리 부족 오류
```bash
# CPU 모드로 실행
python main.py input.png --device cpu
```

## 참고 자료

### 논문 및 기술 문서
- [Table Transformer 논문](https://arxiv.org/abs/2110.00061)
- [Hugging Face 모델 카드](https://huggingface.co/docs/transformers/main/en/model_doc/table-transformer)
- [DocLayNet 데이터셋](https://github.com/DS4SD/DocLayNet)
- [YOLO11 문서](https://docs.ultralytics.com/)

### 프로젝트 문서
- [Table Transformer 사용 가이드](docs/table_transformer_usage.md)
- [DocLayNet 학습 가이드](docs/doclaynet_training_guide.md)
- [Colab 학습 가이드](docs/colab_training_guide.md)

## 라이선스

MIT License

## 기여

이슈 및 풀 리퀘스트를 환영합니다!

## 문의

프로젝트 관련 문의사항은 이슈를 통해 남겨주세요.
