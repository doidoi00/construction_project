# 건설 시방서 표 구조화 프로젝트

YOLO와 OCR을 활용하여 문서의 레이아웃을 분류하고, 건설 시방서의 표를 인식하여 구조화된 JSON 파일로 추출하는 시스템

## 핵심 파이프라인

1. **레이아웃 분류**: YOLO로 문서의 전체 레이아웃 분석 (제목, 본문, 표, 이미지 등)
2. **표 영역 감지**: 분류된 레이아웃 중 표 영역 정밀 탐지
3. **텍스트 추출**: OCR로 감지된 표 영역에서 텍스트 추출
4. **구조 분석**: 텍스트의 위치(x, y 좌표) 기반으로 행/열 구조 파악
5. **JSON 변환**: 파싱된 데이터를 구조화된 JSON 형식으로 저장

## 개발 환경

- **운영체제**: macOS
- **Python**: 3.12
- **하드웨어 제약**: CUDA 사용 불가 (NVIDIA GPU 없음)
- **가속 지원**: Metal Performance Shaders (MPS) 사용 가능
- **패키지 관리**: uv

## 설치 방법

### 1. 저장소 클론
```bash
git clone <repository-url>
cd yolo
```

### 2. 패키지 설치
```bash
# uv를 사용하여 패키지 설치
uv sync

# 또는 수동 설치
uv pip install -r pyproject.toml
```

### 3. 가상환경 활성화 (선택사항)
```bash
source .venv/bin/activate
```

## 사용 방법

### 기본 사용법

```bash
# 단일 이미지 처리
uv run python main.py data/images/page-001.png

# 폴더 내 모든 이미지 처리
uv run python main.py data/images/pdf_images/

# 출력 폴더 지정
uv run python main.py data/images/ -o data/output/

# 커스텀 YOLO 모델 사용
uv run python main.py data/images/ -m models/yolo_weights/custom.pt
```

### 고급 옵션

```bash
# 감지 신뢰도 조정 (기본값: 0.5)
uv run python main.py data/images/ --conf 0.7

# OCR 언어 설정 (기본값: ko,en)
uv run python main.py data/images/ --lang ko,en

# 디바이스 선택 (기본값: 자동 감지)
uv run python main.py data/images/ --device mps

# 도움말 확인
uv run python main.py --help
```

### PDF를 이미지로 변환

```bash
# PDF 파일을 data/raw/pdf/ 폴더에 넣고 실행
uv run python src/convert_pdf.py
```

## 프로젝트 구조

```
yolo/
├── data/
│   ├── raw/              # 원본 PDF 파일
│   │   └── pdf/          # PDF 원본 위치
│   ├── images/           # 처리용 이미지
│   │   └── pdf_images/   # 변환된 PNG 파일
│   ├── doclaynet/        # DocLayNet 데이터셋 (선택)
│   └── output/           # JSON 결과 파일
├── models/
│   ├── yolo_weights/     # YOLO 모델 가중치
│   └── custom_weights/   # 커스텀 학습 모델
├── src/
│   ├── detection.py      # YOLO 레이아웃/표 감지
│   ├── ocr.py            # OCR 텍스트 추출
│   ├── parser.py         # 데이터 구조화 및 JSON 변환
│   ├── utils.py          # 유틸리티 함수
│   └── convert_pdf.py    # PDF to 이미지 변환
├── main.py               # 메인 실행 파일
├── pyproject.toml        # uv 프로젝트 설정
├── CLAUDE.md             # AI 지원 가이드
└── README.md             # 프로젝트 설명
```

## 처리 플로우

```
PDF 문서
    ↓
[이미지 변환] (convert_pdf.py)
    ↓
[레이아웃 감지] (detection.py - YOLO)
    ↓
[표 영역 추출]
    ↓
[텍스트 추출] (ocr.py - EasyOCR)
    ↓
[행/열 구조 분석] (parser.py)
    ↓
[JSON 출력]
```

## 출력 JSON 형식

```json
{
  "document_id": "page-001",
  "metadata": {
    "created_at": "2024-01-01T12:00:00",
    "table_count": 2
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
          ["데이터1", "데이터2", "데이터3"],
          ["데이터4", "데이터5", "데이터6"]
        ],
        "merged_cells": []
      }
    }
  ]
}
```

## 주요 기능

### 1. YOLO 기반 표 감지
- 사전학습된 YOLO11 모델 사용
- MPS(Metal Performance Shaders)로 GPU 가속
- 높은 감지 정확도

### 2. EasyOCR 텍스트 추출
- 한글 + 영어 동시 지원
- 텍스트 위치 좌표 제공
- 표 영역 전용 OCR

### 3. 표 구조 파싱
- Y 좌표로 기반 행 분류
- X 좌표로 기반 열 분류
- 병합 셀 감지 기능

### 4. JSON 출력
- 구조화된 데이터 저장
- 메타데이터 포함
- 추가 처리용 포맷 제공

## 개발 로드맵

### Phase 1: 기본 인프라 구축 ✅
- [x] 프로젝트 구조 설정
- [x] 패키지 설치
- [x] 핵심 모듈 구현

### Phase 2: YOLO 모델 최적화 (진행 예정)
- [ ] DocLayNet 데이터셋 통합
- [ ] 커스텀 모델 학습
- [ ] 표 감지 정확도 향상

### Phase 3: OCR 개선
- [ ] OCR 정확도 테스트
- [ ] 한글 인식 최적화
- [ ] 병합 정확도 개선

### Phase 4: 표 구조 파싱 고도화
- [ ] 복잡한 표 처리
- [ ] 병합 셀 감지 개선
- [ ] 다양한 레이아웃 대응

### Phase 5: 통합 및 최적화
- [ ] 전체 파이프라인 검증
- [ ] 성능 최적화 작업
- [ ] 문서 및 예제 추가

## 주요 기술 스택

- **YOLO**: Ultralytics YOLO11 (문서 레이아웃 감지)
- **OCR**: EasyOCR (한글/영어 텍스트 추출)
- **이미지 처리**: OpenCV, Pillow
- **딥러닝**: PyTorch (MPS 지원)
- **PDF 처리**: PyMuPDF

## macOS/Metal 최적화

이 프로젝트는 macOS 환경에 최적화되어 있습니다:

- ✅ **MPS 지원**: Apple Silicon의 GPU 가속 활용
- ❌ **CUDA 미지원**: NVIDIA GPU 없음
- ⚠️ **성능 고려사항**: MPS 미지원 시 CPU 모드 자동 전환

```python
# 디바이스 자동 감지 코드
from src.utils import get_device
device = get_device()  # "mps" 또는 "cpu" 반환
```

## 문제 해결

### EasyOCR 모델 다운로드 시간
```bash
# 최초 실행 시 모델 다운로드
uv run python -c "import easyocr; easyocr.Reader(['ko', 'en'])"
```

### MPS 설치 오류 발생
```bash
# CPU 모드로 실행
uv run python main.py data/images/ --device cpu
```

### 표 감지 안됨 또는 낮은 정확도
```bash
# 신뢰도 임계값 조정
uv run python main.py data/images/ --conf 0.3
```

## 라이선스

MIT License

## 기여

이슈 및 풀 리퀘스트를 환영합니다!

## 문의

프로젝트 관련 문의사항은 이슈를 통해 남겨주세요.
