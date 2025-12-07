"""
데이터 구조화 및 JSON 변환 모듈

감지된 레이아웃과 OCR 결과를 통합하여:
1. 표 구조 분석 (행/열)
2. 구조화된 JSON 형식으로 변환
3. 파일로 저장
"""

import json
from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime


class TableParser:
    """
    표 구조 파싱 및 JSON 변환 클래스

    주요 기능:
    - OCR 결과를 표 구조로 변환
    - JSON 형식으로 출력
    - 병합 셀 감지 (향후 구현 예정)
    """

    def __init__(self):
        """
        TableParser 초기화

        설명:
            - 현재는 기본 초기화만 수행
            - 향후 설정 옵션 추가 가능 (예: 병합 셀 처리 방법)
        """
        pass

    def parse_table(
        self,
        text_lines: List[List[Dict]],
        bbox: List[float],
        confidence: float
    ) -> Dict[str, Any]:
        """
        행별로 그룹화된 텍스트를 표 구조로 변환합니다.

        Args:
            text_lines: cluster_texts_by_line() 결과
            bbox: 표 영역의 바운딩 박스 [x1, y1, x2, y2]
            confidence: 표 감지 신뢰도

        Returns:
            Dict: 표 구조 데이터
                {
                    "type": "table",
                    "bbox": [x1, y1, x2, y2],
                    "confidence": 0.95,
                    "content": {
                        "rows": 5,
                        "cols": 3,
                        "data": [
                            ["헤더1", "헤더2", "헤더3"],
                            ["데이터1", "데이터2", "데이터3"]
                        ],
                        "merged_cells": []  # 향후 구현
                    }
                }

        설명:
            - text_lines의 각 항목이 표의 한 행에 해당
            - 열 수는 가장 많은 항목을 가진 행을 기준으로 결정
            - 열 수가 다른 행은 빈 셀("")로 패딩
        """
        # 텍스트만 추출하여 2D 배열 생성
        table_data = []
        max_cols = 0

        for line in text_lines:
            row_texts = [item["text"] for item in line]
            table_data.append(row_texts)
            max_cols = max(max_cols, len(row_texts))

        # 열 수를 맞추기 위해 짧은 행에 빈 셀 추가
        for row in table_data:
            while len(row) < max_cols:
                row.append("")

        # 표 구조 생성
        table_structure = {
            "type": "table",
            "bbox": bbox,
            "confidence": confidence,
            "content": {
                "rows": len(table_data),
                "cols": max_cols,
                "data": table_data,
                "merged_cells": []  # TODO: 병합 셀 감지 구현
            }
        }

        return table_structure

    def detect_merged_cells(
        self,
        table_data: List[List[str]]
    ) -> List[Dict]:
        """
        병합된 셀을 감지합니다. (향후 구현 예정)

        Args:
            table_data: 2D 텍스트 배열

        Returns:
            List[Dict]: 병합 셀 정보
                [
                    {
                        "start_row": 0,
                        "start_col": 1,
                        "end_row": 2,
                        "end_col": 1
                    }
                ]

        설명:
            - 빈 셀이 연속되거나 중복된 값이 있는 경우 병합 셀로 추정
            - 현재는 빈 리스트 반환 (placeholder)
            - 향후 구현 시:
              1. 좌표 기반 병합 감지
              2. 빈 셀 패턴 분석
              3. 실제 문서의 선(line) 분석
        """
        if not table_data:
            return []

        rows = len(table_data)
        cols = max(len(row) for row in table_data)
        normalized = [
            row + [""] * (cols - len(row)) if len(row) < cols else row
            for row in table_data
        ]

        merged_cells = []

        for r in range(rows):
            for c in range(cols):
                value = normalized[r][c]
                if not value:
                    continue

                # 이미 동일한 값이 왼쪽/위쪽에 존재하면 해당 셀에서 처리되었으므로 건너뜁니다.
                if c > 0 and normalized[r][c - 1] == value:
                    continue
                if r > 0 and normalized[r - 1][c] == value:
                    continue

                # 가로 확장: 동일 값 혹은 빈 셀이 이어지면 병합 후보
                h_end = c
                while h_end + 1 < cols:
                    candidate = normalized[r][h_end + 1]
                    if candidate in ("", value):
                        h_end += 1
                        continue
                    break

                # 세로 확장: 위에서 찾은 가로 범위가 아래 행에서도 유지되는지 확인
                v_end = r
                while v_end + 1 < rows:
                    next_row = v_end + 1
                    row_valid = True
                    for cc in range(c, h_end + 1):
                        candidate = normalized[next_row][cc]
                        if candidate in ("", value):
                            continue
                        row_valid = False
                        break
                    if not row_valid:
                        break
                    v_end = next_row

                if h_end > c or v_end > r:
                    merged_cells.append({
                        "start_row": r,
                        "start_col": c,
                        "end_row": v_end,
                        "end_col": h_end
                    })

        return merged_cells


class DocumentParser:
    """
    전체 문서 파싱 및 JSON 변환 클래스

    주요 기능:
    - 문서 전체의 레이아웃 정보 통합
    - 여러 표를 하나의 JSON 문서로 변환
    - 메타데이터 추가
    """

    def __init__(self):
        """
        DocumentParser 초기화
        """
        self.table_parser = TableParser()

    def create_document_structure(
        self,
        document_id: str,
        tables: List[Dict],
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        문서 전체의 구조화된 데이터를 생성합니다.

        Args:
            document_id: 문서 식별자 (예: 파일명)
            tables: parse_table() 결과 리스트
            metadata: 추가 메타데이터 (선택)

        Returns:
            Dict: 문서 전체 구조
                {
                    "document_id": "spec_001",
                    "metadata": {
                        "created_at": "2024-01-01T12:00:00",
                        "page_count": 1,
                        "table_count": 3
                    },
                    "layout": [
                        { "type": "table", ... },
                        { "type": "table", ... }
                    ]
                }

        설명:
            - 여러 페이지/여러 표를 하나의 JSON 문서로 통합
            - 메타데이터로 처리 시간, 통계 등 추가 가능
        """
        # 기본 메타데이터 생성
        if metadata is None:
            metadata = {}

        metadata.update({
            "created_at": datetime.now().isoformat(),
            "table_count": len(tables),
        })

        # 문서 구조 생성
        document = {
            "document_id": document_id,
            "metadata": metadata,
            "layout": tables
        }

        return document

    def save_to_json(
        self,
        document: Dict[str, Any],
        output_path: str,
        indent: int = 2,
        ensure_ascii: bool = False
    ) -> None:
        """
        구조화된 데이터를 JSON 파일로 저장합니다.

        Args:
            document: create_document_structure() 반환값
            output_path: 저장 경로 (.json)
            indent: JSON 들여쓰기 크기
            ensure_ascii: False면 한글이 그대로 저장됨

        설명:
            - ensure_ascii=False: 한글을 유니코드 이스케이프 하지 않음
              (True면 "한글" → "\uD55C\uAE00")
            - indent=2: 가독성을 위한 들여쓰기
        """
        # 디렉토리 생성
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # JSON 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(document, f, indent=indent, ensure_ascii=ensure_ascii)

        print(f"✓ JSON 저장: {output_path}")

    def load_from_json(self, json_path: str) -> Dict[str, Any]:
        """
        JSON 파일을 로드합니다.

        Args:
            json_path: JSON 파일 경로

        Returns:
            Dict: 문서 데이터

        설명:
            - 저장된 JSON을 다시 로드하여 후처리 가능
            - 예: 여러 문서 병합, 통계 생성 등
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            document = json.load(f)

        return document


def merge_detection_and_ocr(
    detections: List[Dict],
    ocr_results: Dict[str, List[Dict]]
) -> List[Dict]:
    """
    YOLO 감지 결과와 OCR 결과를 병합합니다.

    Args:
        detections: LayoutDetector.detect_tables() 반환값
        ocr_results: {영역 인덱스: OCR 텍스트 데이터}

    Returns:
        List[Dict]: 감지 + OCR 통합 데이터

    설명:
        - YOLO로 감지된 표 영역과 해당 영역의 OCR 결과를 매칭
        - 각 표 영역에 추출된 텍스트 정보 추가

    예제:
        detections = [{"bbox": [10, 20, 100, 200], ...}]
        ocr_results = {0: [{"text": "헤더1", ...}, ...]}
        → [{"bbox": [10, 20, 100, 200], "texts": [...], ...}]
    """
    merged = []

    for idx, detection in enumerate(detections):
        merged_item = detection.copy()

        # 해당 영역의 OCR 결과 추가
        if idx in ocr_results:
            merged_item["texts"] = ocr_results[idx]
        else:
            merged_item["texts"] = []

        merged.append(merged_item)

    return merged


def validate_table_structure(table: Dict[str, Any]) -> bool:
    """
    표 구조의 유효성을 검증합니다.

    Args:
        table: parse_table() 반환값

    Returns:
        bool: 유효하면 True, 아니면 False

    검증 항목:
        - 필수 필드 존재 여부
        - 행/열 수가 0보다 큼
        - data 배열의 크기가 rows와 일치
        - 각 행의 길이가 cols와 일치

    설명:
        - 파싱 결과의 무결성 확인
        - 잘못된 데이터가 JSON에 저장되는 것을 방지
    """
    try:
        # 필수 필드 확인
        required_fields = ["type", "bbox", "content"]
        for field in required_fields:
            if field not in table:
                print(f"⚠ 필수 필드 누락: {field}")
                return False

        content = table["content"]
        rows = content["rows"]
        cols = content["cols"]
        data = content["data"]

        # 행 수 검증
        if len(data) != rows:
            print(f"⚠ 행 수 불일치: 선언={rows}, 실제={len(data)}")
            return False

        # 열 수 검증
        for i, row in enumerate(data):
            if len(row) != cols:
                print(f"⚠ {i+1}번째 행의 열 수 불일치: 선언={cols}, 실제={len(row)}")
                return False

        return True

    except Exception as e:
        print(f"⚠ 검증 중 오류: {e}")
        return False
