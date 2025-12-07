"""
OCR (Optical Character Recognition) 모듈

EasyOCR을 사용하여:
1. 이미지에서 텍스트 추출
2. 텍스트의 위치(좌표) 정보 함께 반환
3. 한글 및 영어 동시 지원
"""

import os
from typing import List, Dict, Tuple, Optional
from pathlib import Path

import easyocr
import numpy as np
import cv2


class OCRProcessor:
    """
    OCR 처리를 위한 클래스

    주요 기능:
    - EasyOCR을 사용한 텍스트 추출
    - 텍스트의 바운딩 박스 좌표 반환
    - 한글/영어 동시 지원
    """

    def __init__(
        self,
        languages: List[str] = ['ko', 'en'],
        gpu: bool = False
    ):
        """
        OCRProcessor 초기화

        Args:
            languages: 인식할 언어 리스트
                - 'ko': 한글
                - 'en': 영어
                - ['ko', 'en']: 한글+영어 동시 지원
            gpu: GPU 사용 여부
                - macOS에서는 False 권장 (MPS 직접 지원 안 됨)

        설명:
            - EasyOCR은 딥러닝 기반 OCR 라이브러리
            - 첫 실행 시 언어별 모델을 자동 다운로드 (~100MB)
            - 한글 인식이 필요하므로 'ko' 포함 필수
        """
        print(f"OCR 모델 초기화 중 (언어: {', '.join(languages)})")
        self.reader = easyocr.Reader(languages, gpu=gpu)
        print("✓ OCR 모델 준비 완료")

    def extract_text(
        self,
        image_path: str,
        detail: int = 1
    ) -> List[Dict]:
        """
        이미지에서 텍스트를 추출합니다.

        Args:
            image_path: 입력 이미지 경로
            detail: 상세 수준
                - 0: 텍스트만 반환
                - 1: 텍스트 + 좌표 + 신뢰도 반환

        Returns:
            List[Dict]: 추출된 텍스트 정보 리스트
                각 항목은 다음 구조:
                {
                    "text": "추출된 텍스트",
                    "confidence": 0.95,              # 신뢰도 (0~1)
                    "bbox": [[x1,y1], [x2,y2], ...], # 4개 꼭짓점 좌표
                    "center": [cx, cy],              # 중심 좌표
                    "top_left": [x1, y1],            # 좌상단 좌표
                }

        설명:
            - reader.readtext(): EasyOCR의 핵심 메서드
            - 반환값: [(좌표, 텍스트, 신뢰도), ...]
            - 좌표는 4개 점의 리스트 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
              (직사각형이 아닐 수 있어서 4개 점 사용)
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_path}")

        print(f"OCR 처리 중: {Path(image_path).name}")
        results = self.reader.readtext(image_path, detail=detail)

        # 결과를 구조화된 형태로 변환
        text_data = []
        for (bbox, text, confidence) in results:
            # bbox는 4개 꼭짓점 좌표 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            bbox_array = np.array(bbox)

            # 중심 좌표 계산 (행/열 정렬에 사용)
            center_x = float(bbox_array[:, 0].mean())
            center_y = float(bbox_array[:, 1].mean())

            # 좌상단 좌표 (정렬 보조용)
            top_left_x = float(bbox_array[:, 0].min())
            top_left_y = float(bbox_array[:, 1].min())

            text_info = {
                "text": text,
                "confidence": float(confidence),
                "bbox": bbox_array.tolist(),
                "center": [center_x, center_y],
                "top_left": [top_left_x, top_left_y],
            }
            text_data.append(text_info)

        print(f"✓ {len(text_data)}개 텍스트 추출됨")
        return text_data

    def extract_text_from_region(
        self,
        image_path: str,
        bbox: List[float]
    ) -> List[Dict]:
        """
        이미지의 특정 영역에서만 텍스트를 추출합니다.

        Args:
            image_path: 입력 이미지 경로
            bbox: 추출할 영역 [x1, y1, x2, y2]

        Returns:
            List[Dict]: 추출된 텍스트 정보 리스트

        설명:
            - 전체 이미지를 OCR하지 않고 특정 영역만 처리
            - 표 영역을 먼저 감지한 후 해당 영역만 OCR하는 데 사용
            - 처리 속도 향상 및 노이즈 감소
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_path}")

        # 이미지 로드 및 영역 크롭
        image = cv2.imread(image_path)
        x1, y1, x2, y2 = map(int, bbox)
        cropped = image[y1:y2, x1:x2]

        # 임시 파일로 저장 (EasyOCR은 파일 경로나 numpy 배열 입력 가능)
        # numpy 배열 직접 입력 시 메모리 효율적
        results = self.reader.readtext(cropped, detail=1)

        # 좌표를 원본 이미지 기준으로 변환
        text_data = []
        for (bbox_rel, text, confidence) in results:
            # 상대 좌표를 절대 좌표로 변환
            bbox_array = np.array(bbox_rel)
            bbox_array[:, 0] += x1  # x 좌표 오프셋
            bbox_array[:, 1] += y1  # y 좌표 오프셋

            center_x = float(bbox_array[:, 0].mean())
            center_y = float(bbox_array[:, 1].mean())

            top_left_x = float(bbox_array[:, 0].min())
            top_left_y = float(bbox_array[:, 1].min())

            text_info = {
                "text": text,
                "confidence": float(confidence),
                "bbox": bbox_array.tolist(),
                "center": [center_x, center_y],
                "top_left": [top_left_x, top_left_y],
            }
            text_data.append(text_info)

        return text_data

    def visualize_ocr_result(
        self,
        image_path: str,
        text_data: List[Dict],
        output_path: str
    ) -> None:
        """
        OCR 결과를 이미지에 시각화하여 저장합니다.

        Args:
            image_path: 원본 이미지 경로
            text_data: extract_text() 반환값
            output_path: 결과 이미지 저장 경로

        설명:
            - 추출된 텍스트의 바운딩 박스를 이미지에 그림
            - 텍스트 내용도 함께 표시
            - 디버깅 및 검증용
        """
        image = cv2.imread(image_path)

        for item in text_data:
            bbox = np.array(item["bbox"], dtype=np.int32)
            text = item["text"]
            confidence = item["confidence"]

            # 바운딩 박스 그리기 (4개 점을 연결)
            cv2.polylines(image, [bbox], True, (0, 255, 0), 2)

            # 텍스트 표시
            cv2.putText(
                image,
                f"{text} ({confidence:.2f})",
                tuple(bbox[0]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1
            )

        # 저장
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, image)
        print(f"✓ OCR 시각화 저장: {output_path}")


def cluster_texts_by_line(
    text_data: List[Dict],
    y_threshold: float = 10.0
) -> List[List[Dict]]:
    """
    OCR 결과를 Y 좌표 기준으로 행(line)별로 그룹화합니다.

    Args:
        text_data: OCRProcessor.extract_text() 반환값
        y_threshold: 같은 행으로 간주할 Y 좌표 차이 임계값 (픽셀)

    Returns:
        List[List[Dict]]: 행별로 그룹화된 텍스트 리스트
            [[행1의 텍스트들], [행2의 텍스트들], ...]

    설명:
        - 표의 행을 구분하기 위해 Y 좌표로 클러스터링
        - y_threshold 내의 텍스트들은 같은 행으로 간주
        - 각 행 내에서는 X 좌표로 정렬 (좌→우 순서)

    알고리즘:
        1. Y 좌표로 정렬
        2. 인접한 텍스트 간 Y 차이가 threshold 이하면 같은 행
        3. threshold 초과하면 새 행 시작
    """
    if not text_data:
        return []

    # Y 좌표(center_y)로 정렬
    sorted_texts = sorted(text_data, key=lambda x: x["center"][1])

    lines = []
    current_line = [sorted_texts[0]]
    current_y = sorted_texts[0]["center"][1]

    for text in sorted_texts[1:]:
        text_y = text["center"][1]

        # Y 좌표 차이가 임계값 이하면 같은 행
        if abs(text_y - current_y) <= y_threshold:
            current_line.append(text)
        else:
            # 새로운 행 시작
            # 현재 행을 X 좌표로 정렬 후 저장
            current_line.sort(key=lambda x: x["center"][0])
            lines.append(current_line)

            # 새 행 초기화
            current_line = [text]
            current_y = text_y

    # 마지막 행 추가
    if current_line:
        current_line.sort(key=lambda x: x["center"][0])
        lines.append(current_line)

    print(f"✓ {len(lines)}개 행으로 그룹화됨")
    return lines


def extract_table_text(
    text_lines: List[List[Dict]]
) -> List[List[str]]:
    """
    행별로 그룹화된 텍스트에서 문자열만 추출하여 2D 배열로 반환합니다.

    Args:
        text_lines: cluster_texts_by_line() 반환값

    Returns:
        List[List[str]]: 표 형태의 텍스트 배열
            [
                ["헤더1", "헤더2", "헤더3"],
                ["데이터1", "데이터2", "데이터3"],
                ...
            ]

    설명:
        - JSON 출력을 위한 단순화된 표 구조
        - 좌표 정보는 제거하고 텍스트만 추출
    """
    table_data = []

    for line in text_lines:
        row_texts = [item["text"] for item in line]
        table_data.append(row_texts)

    return table_data
