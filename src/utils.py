"""
유틸리티 함수 모음
- 디바이스 설정 (MPS/CPU)
- 이미지 로드 및 전처리
- 결과 시각화
"""

import os
import torch
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional


def get_device() -> str:
    """
    실행 환경에 맞는 디바이스를 반환합니다.

    macOS에서는 Metal Performance Shaders (MPS)를 사용하고,
    지원하지 않는 경우 CPU를 사용합니다.

    Returns:
        str: "mps" 또는 "cpu"

    설명:
        - torch.backends.mps.is_available(): Apple Silicon의 GPU 가속 지원 여부 확인
        - MPS는 M1/M2 칩에서 딥러닝 모델을 빠르게 실행할 수 있게 해줌
        - CUDA는 사용 불가 (NVIDIA GPU 없음)
    """
    if torch.backends.mps.is_available():
        print("✓ Metal Performance Shaders (MPS) 사용 가능")
        return "mps"
    else:
        print("⚠ MPS 사용 불가, CPU 모드로 실행")
        return "cpu"


def load_image(image_path: str) -> np.ndarray:
    """
    이미지 파일을 로드합니다.

    Args:
        image_path: 이미지 파일 경로

    Returns:
        np.ndarray: BGR 형식의 이미지 배열 (OpenCV 기본 형식)

    Raises:
        FileNotFoundError: 이미지 파일이 존재하지 않을 때
        ValueError: 이미지 로드에 실패했을 때

    설명:
        - cv2.imread()는 이미지를 numpy 배열로 읽어옴
        - BGR 순서로 채널이 저장됨 (RGB 아님!)
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"이미지 로드 실패: {image_path}")

    return image


def draw_boxes(
    image: np.ndarray,
    boxes: List[List[float]],
    labels: List[str],
    scores: List[float],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    """
    이미지에 바운딩 박스와 레이블을 그립니다.

    Args:
        image: 원본 이미지 (numpy 배열)
        boxes: 바운딩 박스 좌표 리스트 [[x1, y1, x2, y2], ...]
        labels: 각 박스의 레이블 리스트
        scores: 각 박스의 신뢰도 점수 리스트 (0~1)
        color: 박스 색상 (B, G, R) 형식
        thickness: 박스 선 두께

    Returns:
        np.ndarray: 박스가 그려진 이미지

    설명:
        - cv2.rectangle(): 사각형 그리기
        - cv2.putText(): 텍스트 추가
        - 원본 이미지를 수정하지 않기 위해 copy() 사용
    """
    result_image = image.copy()

    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = map(int, box)

        # 바운딩 박스 그리기
        cv2.rectangle(result_image, (x1, y1), (x2, y2), color, thickness)

        # 레이블과 점수 텍스트 작성
        text = f"{label}: {score:.2f}"
        cv2.putText(
            result_image,
            text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            thickness
        )

    return result_image


def save_result(
    image: np.ndarray,
    output_path: str,
    create_dir: bool = True
) -> None:
    """
    결과 이미지를 저장합니다.

    Args:
        image: 저장할 이미지
        output_path: 저장 경로
        create_dir: 디렉토리가 없으면 생성할지 여부

    설명:
        - Path().parent: 파일의 부모 디렉토리 경로 추출
        - mkdir(parents=True): 중간 디렉토리까지 모두 생성
        - exist_ok=True: 이미 존재해도 에러 발생하지 않음
    """
    if create_dir:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(output_path, image)
    print(f"✓ 결과 저장: {output_path}")


def create_output_path(
    input_path: str,
    output_dir: str,
    suffix: str = "_result"
) -> str:
    """
    입력 파일 경로로부터 출력 파일 경로를 생성합니다.

    Args:
        input_path: 입력 파일 경로 (예: "data/images/page-001.png")
        output_dir: 출력 디렉토리 (예: "data/output")
        suffix: 파일명에 추가할 접미사

    Returns:
        str: 출력 파일 경로 (예: "data/output/page-001_result.png")

    예제:
        >>> create_output_path("data/images/page-001.png", "data/output")
        "data/output/page-001_result.png"
    """
    input_file = Path(input_path)
    filename = input_file.stem + suffix + input_file.suffix
    return str(Path(output_dir) / filename)
