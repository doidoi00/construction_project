"""
YOLO 기반 문서 레이아웃 및 표 감지 모듈

이 모듈은 YOLO 모델을 사용하여:
1. 문서 이미지에서 레이아웃 요소 감지 (표, 텍스트, 이미지 등)
2. 특히 표(Table) 영역을 정밀하게 탐지
3. 감지된 영역의 좌표(바운딩 박스)를 반환
"""

import os
from typing import List, Dict, Tuple, Optional
from pathlib import Path

import torch
import numpy as np
from ultralytics import YOLO

from src.utils import get_device


class LayoutDetector:
    """
    문서 레이아웃 감지를 위한 YOLO 모델 래퍼 클래스

    주요 기능:
    - YOLO 모델 로드 및 초기화
    - 문서 이미지에서 레이아웃 요소 감지
    - 표 영역만 필터링
    - 감지 결과를 구조화된 형태로 반환
    """

    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        device: Optional[str] = None,
        confidence_threshold: float = 0.5
    ):
        """
        LayoutDetector 초기화

        Args:
            model_path: YOLO 모델 가중치 파일 경로
                - "yolo11n.pt": 사전학습된 YOLO11 nano 모델 (가벼움)
                - "yolo11s.pt", "yolo11m.pt" 등: 더 큰 모델 (정확도 높음)
                - 커스텀 학습 모델 경로도 가능
            device: 실행 디바이스 ("mps", "cpu" 등)
                - None이면 자동으로 최적 디바이스 선택
            confidence_threshold: 감지 신뢰도 임계값 (0~1)
                - 이 값보다 낮은 신뢰도의 감지는 무시됨

        설명:
            - YOLO 모델은 객체 감지(Object Detection)를 수행
            - 이미지 → 바운딩 박스 + 클래스 + 신뢰도 출력
        """
        self.device = device if device else get_device()
        self.confidence_threshold = confidence_threshold

        print(f"모델 로드 중: {model_path}")
        self.model = YOLO(model_path)

        # 모델을 지정된 디바이스로 이동
        # MPS 사용 시 GPU 가속으로 빠른 추론 가능
        self.model.to(self.device)
        print(f"✓ 모델이 {self.device}에 로드되었습니다")

        # 모델 클래스 정보 출력 (디버깅)
        if hasattr(self.model, 'names'):
            print(f"✓ 모델 클래스: {list(self.model.names.values())}")

    def detect(
        self,
        image_path: str,
        target_classes: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        이미지에서 레이아웃 요소를 감지합니다.

        Args:
            image_path: 입력 이미지 경로
            target_classes: 필터링할 클래스 목록 (예: ["table"])
                - None이면 모든 클래스 반환
                - ["table"]이면 표만 반환

        Returns:
            List[Dict]: 감지된 객체 리스트
                각 항목은 다음 구조:
                {
                    "class": "table",              # 클래스명
                    "confidence": 0.95,            # 신뢰도 (0~1)
                    "bbox": [x1, y1, x2, y2],     # 바운딩 박스 좌표
                    "bbox_norm": [x1, y1, x2, y2] # 정규화된 좌표 (0~1)
                }

        설명:
            - self.model.predict(): YOLO 모델로 추론 실행
            - 결과는 ultralytics의 Results 객체로 반환됨
            - boxes.xyxy: 바운딩 박스 좌표 [x1, y1, x2, y2]
            - boxes.conf: 신뢰도 점수
            - boxes.cls: 클래스 ID
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_path}")

        # YOLO 모델로 추론 실행
        # verbose=False: 진행 메시지 숨김
        results = self.model.predict(
            source=image_path,
            device=self.device,
            conf=self.confidence_threshold,
            verbose=False
        )

        # 결과 파싱
        detections = []
        result = results[0]  # 첫 번째 이미지 결과 (배치 처리 시 여러 개 가능)

        # 감지된 각 객체에 대해 처리
        for box in result.boxes:
            # 클래스명 가져오기
            class_id = int(box.cls[0])
            class_name = result.names[class_id]

            # target_classes 필터링
            if target_classes and class_name not in target_classes:
                continue

            # 바운딩 박스 좌표 추출
            xyxy = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
            confidence = float(box.conf[0])

            # 정규화된 좌표 (이미지 크기 대비 0~1 범위)
            img_height, img_width = result.orig_shape
            xyxy_norm = [
                xyxy[0] / img_width,
                xyxy[1] / img_height,
                xyxy[2] / img_width,
                xyxy[3] / img_height,
            ]

            detection = {
                "class": class_name,
                "confidence": confidence,
                "bbox": xyxy.tolist(),
                "bbox_norm": xyxy_norm,
            }
            detections.append(detection)

        print(f"✓ {len(detections)}개 객체 감지됨 (이미지: {Path(image_path).name})")
        return detections

    def detect_tables(self, image_path: str) -> List[Dict]:
        """
        이미지에서 표(Table) 영역만 감지합니다.

        Args:
            image_path: 입력 이미지 경로

        Returns:
            List[Dict]: 감지된 표 리스트

        설명:
            - detect() 메서드를 호출하되 target_classes=["Table"]로 필터링
            - DocLayNet은 대문자 "Table" 클래스 사용
        """
        # DocLayNet 모델은 "Table" (대문자) 클래스를 사용
        # 소문자/대문자 모두 지원하도록 수정
        all_detections = self.detect(image_path, target_classes=None)

        # 모든 감지된 클래스 출력 (디버깅)
        detected_classes = set([d["class"] for d in all_detections])
        if detected_classes:
            print(f"  감지된 클래스: {detected_classes}")

        # "Table" 또는 "table" 클래스만 필터링
        table_detections = [
            d for d in all_detections
            if d["class"].lower() == "table"
        ]

        if not table_detections and all_detections:
            print(f"  ⚠️  표가 감지되지 않았습니다. 다른 클래스가 감지되었는지 확인하세요.")
            print(f"  💡 신뢰도 임계값을 낮춰보세요: --conf 0.3")

        return table_detections

    def batch_detect(
        self,
        image_dir: str,
        output_dir: Optional[str] = None,
        target_classes: Optional[List[str]] = None
    ) -> Dict[str, List[Dict]]:
        """
        디렉토리 내 모든 이미지를 배치로 처리합니다.

        Args:
            image_dir: 이미지 디렉토리 경로
            output_dir: 결과 저장 디렉토리 (None이면 저장 안 함)
            target_classes: 필터링할 클래스 목록

        Returns:
            Dict[str, List[Dict]]: {파일명: 감지결과} 형태의 딕셔너리

        설명:
            - 디렉토리 내 .png, .jpg, .jpeg 파일을 모두 처리
            - 각 파일의 감지 결과를 딕셔너리로 반환
        """
        image_dir_path = Path(image_dir)
        if not image_dir_path.exists():
            raise FileNotFoundError(f"디렉토리를 찾을 수 없습니다: {image_dir}")

        # 이미지 파일 목록 수집
        image_extensions = [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]
        image_files = [
            f for f in image_dir_path.iterdir()
            if f.suffix in image_extensions
        ]

        if not image_files:
            print(f"⚠ {image_dir}에 이미지 파일이 없습니다")
            return {}

        print(f"배치 처리 시작: {len(image_files)}개 파일")

        results = {}
        for image_file in image_files:
            detections = self.detect(str(image_file), target_classes)
            results[image_file.name] = detections

        print(f"✓ 배치 처리 완료")
        return results


def crop_detected_regions(
    image_path: str,
    detections: List[Dict],
    output_dir: str
) -> List[str]:
    """
    감지된 영역을 원본 이미지에서 잘라내어 저장합니다.

    Args:
        image_path: 원본 이미지 경로
        detections: detect() 메서드의 반환값
        output_dir: 잘라낸 이미지 저장 디렉토리

    Returns:
        List[str]: 저장된 파일 경로 리스트

    설명:
        - 표 영역을 잘라내어 별도 이미지로 저장
        - OCR 처리 전에 표 영역만 추출하는 데 사용
    """
    import cv2

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_path}")

    image = cv2.imread(image_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cropped_paths = []
    base_name = Path(image_path).stem

    for idx, detection in enumerate(detections):
        bbox = detection["bbox"]
        x1, y1, x2, y2 = map(int, bbox)

        # 이미지 크롭
        cropped = image[y1:y2, x1:x2]

        # 저장 경로 생성
        class_name = detection["class"]
        filename = f"{base_name}_{class_name}_{idx:03d}.png"
        save_path = output_path / filename

        cv2.imwrite(str(save_path), cropped)
        cropped_paths.append(str(save_path))

    print(f"✓ {len(cropped_paths)}개 영역 저장: {output_dir}")
    return cropped_paths
