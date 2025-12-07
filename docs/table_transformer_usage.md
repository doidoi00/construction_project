# Table Transformer 사용 가이드

## 개요

Table Transformer는 Microsoft가 개발한 표 구조 인식 전문 모델로, DETR(DEtection TRansformer) 아키텍처를 기반으로 합니다. 본 프로젝트에서는 YOLO로 감지한 표 영역의 내부 구조를 정밀하게 분석하는 데 사용됩니다.

## 주요 기능

### 1. 표 구조 인식
- **행(Row) 감지**: 표의 각 행 경계를 자동으로 감지
- **열(Column) 감지**: 표의 각 열 경계를 자동으로 감지
- **셀(Cell) 그리드 생성**: 행과 열의 교차점으로 개별 셀 영역 계산
- **병합 셀 인식**: 여러 셀이 병합된 영역 자동 감지

### 2. 기존 방식과의 차이

| 기능 | 기존 방식 (OCR 좌표 기반) | Table Transformer |
|------|-------------------------|-------------------|
| 행 감지 | Y 좌표 클러스터링 | Transformer 모델 직접 감지 |
| 열 감지 | X 좌표 정렬 | Transformer 모델 직접 감지 |
| 병합 셀 | 휴리스틱 추정 | 모델이 직접 감지 |
| 복잡한 표 | 정확도 낮음 (60-85%) | 정확도 높음 (추정 90%+) |
| 불규칙 간격 | 오류 발생 가능 | 강건한 처리 |

## 사용 방법

### 기본 사용법

```bash
# Table Transformer 활성화 (기본값)
python main.py data/test_table.png -o data/output

# 디렉토리 일괄 처리
python main.py data/images/ -o data/output
```

### 고급 옵션

```bash
# YOLO 및 Table Transformer 신뢰도 조정
python main.py data/test_table.png \
  --conf 0.5 \              # YOLO 신뢰도 임계값
  --structure-conf 0.7      # Table Transformer 신뢰도 임계값

# 기존 방식 사용 (Table Transformer 비활성화)
python main.py data/test_table.png --no-structure

# 디바이스 지정
python main.py data/test_table.png --device mps  # Apple Silicon
python main.py data/test_table.png --device cpu  # CPU
```

## 출력 형식

Table Transformer 사용 시 다음과 같은 구조화된 JSON이 생성됩니다:

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

### 병합 셀 표현

병합 셀은 `merged_cells` 배열에 다음 형식으로 저장됩니다:

```json
{
  "start_row": 0,    // 병합 시작 행 (0-based index)
  "start_col": 1,    // 병합 시작 열 (0-based index)
  "end_row": 0,      // 병합 끝 행 (inclusive)
  "end_col": 2       // 병합 끝 열 (inclusive)
}
```

예: `start_row=0, start_col=1, end_row=0, end_col=2`는 첫 번째 행의 2-3번째 열이 병합되었음을 의미

## 처리 흐름

### 전체 파이프라인

```
입력 이미지
    ↓
[1] YOLO 표 감지
    ↓
표 영역 크롭
    ↓
[2] Table Transformer 구조 인식
    ├─ 행 감지
    ├─ 열 감지
    └─ 병합 셀 감지
    ↓
[3] 셀 그리드 생성
    ↓
[4] OCR 텍스트 추출
    ↓
[5] 텍스트 → 셀 매핑
    ↓
[6] JSON 출력
```

### 세부 처리 과정

1. **표 영역 감지** (YOLO)
   - 입력: 문서 이미지
   - 출력: 표 영역 바운딩 박스 `[x1, y1, x2, y2]`

2. **표 구조 인식** (Table Transformer)
   - 입력: 크롭된 표 이미지
   - 출력:
     - `rows`: 각 행의 바운딩 박스 리스트
     - `columns`: 각 열의 바운딩 박스 리스트
     - `spanning_cells`: 병합 셀 바운딩 박스 리스트

3. **셀 그리드 생성**
   - 입력: 행/열 정보
   - 처리: 행과 열의 교차점 계산
   - 출력: `[rows][cols]` 2D 그리드

4. **OCR 텍스트 추출**
   - 입력: 표 이미지
   - 출력: 텍스트 + 바운딩 박스 리스트

5. **텍스트 매핑**
   - 입력: OCR 결과 + 셀 그리드
   - 처리: 각 텍스트의 중심점이 속하는 셀 찾기
   - 출력: 텍스트가 채워진 2D 배열

## 테스트

### 기본 테스트

```bash
# Table Transformer 모듈 단독 테스트
python test_table_structure.py

# 전체 파이프라인 테스트 정보
python test_table_structure.py --full
```

### Python 코드에서 사용

```python
from src.table_structure import TableStructureRecognizer
from src.utils import get_device

# 초기화
device = get_device()
recognizer = TableStructureRecognizer(
    device=device,
    confidence_threshold=0.7
)

# 표 구조 인식
structure = recognizer.recognize_structure("table_image.png")

print(f"감지된 행: {len(structure['rows'])}개")
print(f"감지된 열: {len(structure['columns'])}개")
print(f"병합 셀: {len(structure['spanning_cells'])}개")

# 셀 그리드 생성
cell_grid = recognizer.create_cell_grid(structure)
print(f"셀 그리드 크기: {len(cell_grid)}x{len(cell_grid[0])}")
```

## 성능 최적화

### macOS/MPS 환경

Table Transformer는 PyTorch 기반이므로 Apple Silicon의 MPS를 활용할 수 있습니다:

```python
# 자동으로 MPS 감지 및 사용
device = get_device()  # "mps" 또는 "cpu"

# 명시적으로 지정
recognizer = TableStructureRecognizer(device="mps")
```

### 신뢰도 임계값 조정

- **높은 정확도 필요**: `--structure-conf 0.8` (더 확실한 감지만 사용)
- **높은 재현율 필요**: `--structure-conf 0.5` (더 많은 감지 포함)
- **권장값**: `0.7` (정확도와 재현율 균형)

## 문제 해결

### Q1: 행/열이 제대로 감지되지 않아요
**A**: 신뢰도 임계값을 낮춰보세요
```bash
python main.py input.png --structure-conf 0.5
```

### Q2: 너무 많은 행/열이 감지돼요
**A**: 신뢰도 임계값을 높이거나 이미지 품질을 개선하세요
```bash
python main.py input.png --structure-conf 0.8
```

### Q3: 병합 셀이 인식되지 않아요
**A**: Table Transformer는 `table spanning cell` 클래스로 병합 셀을 감지합니다. 신뢰도를 낮추거나 이미지 해상도를 높여보세요.

### Q4: OCR 텍스트가 잘못된 셀에 매핑돼요
**A**:
1. Table Transformer 구조 인식 결과를 확인
2. OCR 바운딩 박스 정확도 확인
3. 필요시 수동 후처리 로직 추가

### Q5: 메모리 부족 오류가 발생해요
**A**: CPU 모드로 실행하거나 이미지 크기를 줄이세요
```bash
python main.py input.png --device cpu
```

## 참고 자료

- [Table Transformer 논문](https://arxiv.org/abs/2110.00061)
- [Hugging Face 모델 카드](https://huggingface.co/docs/transformers/main/en/model_doc/table-transformer)
- [PubTables-1M 데이터셋](https://arxiv.org/abs/2110.00061)
- [DETR 아키텍처](https://arxiv.org/abs/2005.12872)
