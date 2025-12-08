"""
Table Transformer 기반 표 구조 인식 모듈

이 모듈은 Microsoft의 Table Transformer를 사용하여:
1. 표 영역 내부의 정밀한 구조 분석
2. 개별 셀 경계 감지
3. 행/열 구조 인식
4. 병합 셀 자동 감지

Table Transformer는 DETR(DEtection TRansformer) 기반으로:
- Table Detection: 문서에서 표 영역 탐지
- Table Structure Recognition: 표 내부의 행/열/셀 구조 인식
"""

import os
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, TableTransformerForObjectDetection

from src.utils import get_device


class TableStructureRecognizer:
    """
    Table Transformer를 사용한 표 구조 인식 클래스

    주요 기능:
    - 표 내부의 행(row), 열(column), 셀(cell) 감지
    - 병합 셀 자동 인식
    - 계층적 표 구조 파싱

    사용 모델:
    - microsoft/table-transformer-structure-recognition
      (PubTables-1M 데이터셋으로 학습된 모델)
    """

    def __init__(
        self,
        model_name: str = "microsoft/table-transformer-structure-recognition",
        device: Optional[str] = None,
        confidence_threshold: float = 0.7
    ):
        """
        TableStructureRecognizer 초기화

        Args:
            model_name: Hugging Face 모델 이름
                - "microsoft/table-transformer-structure-recognition": 구조 인식 (권장)
                - "microsoft/table-transformer-detection": 표 탐지용
            device: 실행 디바이스 ("mps", "cpu" 등)
                - None이면 자동으로 최적 디바이스 선택
            confidence_threshold: 감지 신뢰도 임계값 (0~1)

        설명:
            - AutoImageProcessor: 이미지 전처리 (리사이징, 정규화 등)
            - TableTransformerForObjectDetection: DETR 기반 객체 감지 모델
            - 모델은 표 내부의 여러 객체를 감지:
              * table row (행)
              * table column (열)
              * table (전체 표)
              * table spanning cell (병합 셀)
        """
        self.device = device if device else get_device()
        self.confidence_threshold = confidence_threshold
        self.model_name = model_name

        print(f"Table Transformer 모델 로드 중: {model_name}")

        # 이미지 전처리기 로드
        # AutoImageProcessor: 모델에 맞는 이미지 전처리 자동 선택
        self.image_processor = AutoImageProcessor.from_pretrained(model_name)

        # Table Transformer 모델 로드
        # TableTransformerForObjectDetection: DETR 기반 객체 감지 모델
        self.model = TableTransformerForObjectDetection.from_pretrained(model_name)

        # 모델을 지정된 디바이스로 이동
        self.model.to(self.device)
        self.model.eval()  # 평가 모드 (추론 시 필수)

        print(f"✓ Table Transformer가 {self.device}에 로드되었습니다")

    def recognize_structure(
        self,
        image_path: str,
        bbox: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        표 이미지에서 구조를 인식합니다.

        Args:
            image_path: 표 이미지 경로 (YOLO로 크롭된 표 영역 또는 전체 이미지)
            bbox: 표 영역의 바운딩 박스 [x1, y1, x2, y2] (선택)
                - 전체 이미지에서 특정 영역만 처리할 때 사용

        Returns:
            Dict: 표 구조 정보
                {
                    "rows": [...],      # 감지된 행 리스트
                    "columns": [...],   # 감지된 열 리스트
                    "cells": [...],     # 감지된 셀 리스트
                    "image_size": (width, height)
                }

        처리 과정:
            1. 이미지 로드 및 전처리
            2. Table Transformer로 추론
            3. 결과 후처리 (NMS, 필터링 등)
            4. 행/열/셀 정보 추출
        """
        # 이미지 로드
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_path}")

        image = Image.open(image_path).convert("RGB")

        # bbox가 주어진 경우 해당 영역만 크롭
        if bbox is not None:
            x1, y1, x2, y2 = map(int, bbox)
            image = image.crop((x1, y1, x2, y2))

        image_width, image_height = image.size

        # 이미지 전처리
        # image_processor: 모델 입력 형식에 맞게 변환 (텐서화, 정규화 등)
        # return_tensors="pt": PyTorch 텐서로 반환
        inputs = self.image_processor(images=image, return_tensors="pt")

        # 입력을 디바이스로 이동
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 추론 실행
        with torch.no_grad():  # 그래디언트 계산 비활성화 (메모리 절약)
            outputs = self.model(**inputs)
            
        # 결과 후처리
        # post_process_object_detection: 바운딩 박스 디코딩 및 NMS 적용
        # target_sizes: 원본 이미지 크기로 좌표 변환
        target_sizes = torch.tensor([[image_height, image_width]]).to(self.device)
        results = self.image_processor.post_process_object_detection(
            outputs,
            threshold=self.confidence_threshold,
            target_sizes=target_sizes
        )[0]

        # 결과 파싱
        structure = self._parse_structure_results(
            results,
            image_width,
            image_height
        )

        return structure

    def _parse_structure_results(
        self,
        results: Dict,
        image_width: int,
        image_height: int
    ) -> Dict[str, Any]:
        """
        Table Transformer 결과를 파싱합니다.

        Args:
            results: post_process_object_detection 결과
            image_width: 이미지 너비
            image_height: 이미지 높이

        Returns:
            Dict: 파싱된 구조 정보

        설명:
            - results에는 boxes, scores, labels가 포함됨
            - boxes: [x1, y1, x2, y2] 형식의 바운딩 박스
            - scores: 신뢰도 점수
            - labels: 클래스 ID

            클래스 매핑 (PubTables-1M 기준):
            - 0: table
            - 1: table column
            - 2: table row
            - 3: table column header
            - 4: table projected row header
            - 5: table spanning cell
        """
        # 클래스 ID to 이름 매핑
        id2label = self.model.config.id2label

        rows = []
        columns = []
        cells = []
        spanning_cells = []

        # 각 감지된 객체 처리
        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        labels = results["labels"].cpu().numpy()

        for box, score, label in zip(boxes, scores, labels):
            class_name = id2label[int(label)]
            bbox = box.tolist()  # [x1, y1, x2, y2]

            # 정규화된 좌표 계산 (0~1 범위)
            bbox_norm = [
                bbox[0] / image_width,
                bbox[1] / image_height,
                bbox[2] / image_width,
                bbox[3] / image_height,
            ]

            detection = {
                "bbox": bbox,
                "bbox_norm": bbox_norm,
                "confidence": float(score),
                "class": class_name
            }

            # 클래스별로 분류 (PubTables-1M 클래스 기준)
            # 0: table (무시)
            # 1: table column
            # 2: table row
            # 3: table column header (헤더 행으로 처리)
            # 4: table projected row header (행으로 처리)
            # 5: table spanning cell
            label_id = int(label)
            
            if label_id == 1:  # table column
                columns.append(detection)
            elif label_id == 2:  # table row
                rows.append(detection)
            elif label_id == 3:  # table column header (헤더도 행으로 처리)
                detection["is_header"] = True
                rows.append(detection)
            elif label_id == 4:  # table projected row header (행으로 처리)
                detection["is_projected_header"] = True
                rows.append(detection)
            elif label_id == 5:  # table spanning cell
                spanning_cells.append(detection)
            # label_id == 0 (table)은 무시
            # 일반 셀도 추가 가능 (필요시)

        # 행을 Y 좌표 기준으로 정렬 (위에서 아래로)
        rows = sorted(rows, key=lambda r: r["bbox"][1])

        # 열을 X 좌표 기준으로 정렬 (왼쪽에서 오른쪽으로)
        columns = sorted(columns, key=lambda c: c["bbox"][0])

        structure = {
            "rows": rows,
            "columns": columns,
            "spanning_cells": spanning_cells,
            "image_size": (image_width, image_height)
        }

        print(f"✓ 구조 인식: {len(rows)}개 행, {len(columns)}개 열, {len(spanning_cells)}개 병합 셀")

        return structure

    def create_cell_grid(
        self,
        structure: Dict[str, Any]
    ) -> List[List[Dict]]:
        """
        행/열 정보로부터 셀 그리드를 생성합니다.

        Args:
            structure: recognize_structure() 반환값

        Returns:
            List[List[Dict]]: 2D 셀 그리드
                각 셀은 다음 형식:
                {
                    "row": 0,
                    "col": 1,
                    "bbox": [x1, y1, x2, y2],
                    "is_spanning": False
                }

        설명:
            - 행과 열의 교차점을 계산하여 개별 셀 영역 생성
            - 병합 셀 정보를 반영
            - OCR 결과를 매핑하기 위한 그리드 구조 제공
        """
        rows = structure["rows"]
        columns = structure["columns"]
        spanning_cells = structure["spanning_cells"]

        if not rows or not columns:
            print("⚠ 행 또는 열이 감지되지 않았습니다.")
            return []

        # 셀 그리드 초기화
        num_rows = len(rows)
        num_cols = len(columns)
        grid = [[None for _ in range(num_cols)] for _ in range(num_rows)]

        # 각 행-열 교차점에 셀 생성
        for i, row in enumerate(rows):
            for j, col in enumerate(columns):
                # 교차 영역 계산
                x1 = col["bbox"][0]
                y1 = row["bbox"][1]
                x2 = col["bbox"][2]
                y2 = row["bbox"][3]

                cell = {
                    "row": i,
                    "col": j,
                    "bbox": [x1, y1, x2, y2],
                    "is_spanning": False,
                    "row_span": 1,
                    "col_span": 1
                }

                grid[i][j] = cell

        # 병합 셀 처리
        for spanning_cell in spanning_cells:
            sc_bbox = spanning_cell["bbox"]

            # 병합 셀이 포함하는 행/열 범위 찾기
            start_row = None
            end_row = None
            start_col = None
            end_col = None

            for i, row in enumerate(rows):
                if self._bbox_overlap(sc_bbox, row["bbox"], axis="y"):
                    if start_row is None:
                        start_row = i
                    end_row = i

            for j, col in enumerate(columns):
                if self._bbox_overlap(sc_bbox, col["bbox"], axis="x"):
                    if start_col is None:
                        start_col = j
                    end_col = j

            # 병합 셀 표시
            if start_row is not None and start_col is not None:
                for i in range(start_row, end_row + 1):
                    for j in range(start_col, end_col + 1):
                        if grid[i][j] is not None:
                            grid[i][j]["is_spanning"] = True
                            if i == start_row and j == start_col:
                                # 병합 셀의 시작 지점에 span 정보 저장
                                grid[i][j]["row_span"] = end_row - start_row + 1
                                grid[i][j]["col_span"] = end_col - start_col + 1

        print(f"✓ 셀 그리드 생성: {num_rows}x{num_cols}")

        return grid

    def _bbox_overlap(
        self,
        bbox1: List[float],
        bbox2: List[float],
        axis: str = "both"
    ) -> bool:
        """
        두 바운딩 박스가 겹치는지 확인합니다.

        Args:
            bbox1: [x1, y1, x2, y2]
            bbox2: [x1, y1, x2, y2]
            axis: 검사할 축 ("x", "y", "both")

        Returns:
            bool: 겹치면 True
        """
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2

        if axis == "x":
            return not (x1_max < x2_min or x2_max < x1_min)
        elif axis == "y":
            return not (y1_max < y2_min or y2_max < y1_min)
        else:  # both
            x_overlap = not (x1_max < x2_min or x2_max < x1_min)
            y_overlap = not (y1_max < y2_min or y2_max < y1_min)
            return x_overlap and y_overlap


def match_ocr_to_cells(
    ocr_results: List[Dict],
    cell_grid: List[List[Dict]]
) -> List[List[str]]:
    """
    OCR 결과를 셀 그리드에 매핑합니다.

    Args:
        ocr_results: OCR 텍스트 결과
            [{"text": "텍스트", "bbox": [x1, y1, x2, y2]}, ...]
        cell_grid: create_cell_grid() 반환값

    Returns:
        List[List[str]]: 텍스트가 채워진 2D 표 데이터

    설명:
        - 각 OCR 텍스트의 중심점이 어느 셀에 속하는지 계산
        - 해당 셀에 텍스트 할당
        - 여러 텍스트가 같은 셀에 속하면 공백으로 연결
    """
    if not cell_grid:
        return []

    num_rows = len(cell_grid)
    num_cols = len(cell_grid[0]) if cell_grid else 0

    # 텍스트 그리드 초기화
    text_grid = [["" for _ in range(num_cols)] for _ in range(num_rows)]

    # 각 OCR 결과를 셀에 매핑
    for ocr_item in ocr_results:
        text = ocr_item["text"]
        bbox = ocr_item["bbox"]

        # 텍스트의 중심점 계산
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2

        # 중심점이 속하는 셀 찾기
        for i, row in enumerate(cell_grid):
            for j, cell in enumerate(row):
                if cell is None:
                    continue

                cell_bbox = cell["bbox"]
                if (cell_bbox[0] <= center_x <= cell_bbox[2] and
                    cell_bbox[1] <= center_y <= cell_bbox[3]):
                    # 셀에 텍스트 추가 (이미 있으면 공백으로 연결)
                    if text_grid[i][j]:
                        text_grid[i][j] += " " + text
                    else:
                        text_grid[i][j] = text
                    break

    return text_grid


def create_structured_table(
    cell_grid: List[List[Dict]],
    text_grid: List[List[str]]
) -> Dict[str, Any]:
    """
    셀 그리드와 텍스트 그리드를 통합하여 구조화된 표 데이터를 생성합니다.

    Args:
        cell_grid: create_cell_grid() 반환값
        text_grid: match_ocr_to_cells() 반환값

    Returns:
        Dict: 구조화된 표 데이터
            {
                "rows": 5,
                "cols": 3,
                "data": [["헤더1", "헤더2", "헤더3"], ...],
                "merged_cells": [
                    {
                        "start_row": 0,
                        "start_col": 1,
                        "end_row": 0,
                        "end_col": 2
                    }
                ]
            }
    """
    if not cell_grid or not text_grid:
        return {
            "rows": 0,
            "cols": 0,
            "data": [],
            "merged_cells": []
        }

    num_rows = len(cell_grid)
    num_cols = len(cell_grid[0]) if cell_grid else 0

    # 병합 셀 정보 추출
    merged_cells = []
    for i, row in enumerate(cell_grid):
        for j, cell in enumerate(row):
            if cell is None:
                continue

            # 병합 셀이고 시작 지점인 경우
            if cell["is_spanning"] and cell["row_span"] > 1 or cell["col_span"] > 1:
                merged_cells.append({
                    "start_row": i,
                    "start_col": j,
                    "end_row": i + cell["row_span"] - 1,
                    "end_col": j + cell["col_span"] - 1
                })

    structured_table = {
        "rows": num_rows,
        "cols": num_cols,
        "data": text_grid,
        "merged_cells": merged_cells
    }

    return structured_table
