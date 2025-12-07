"""
건설 시방서 표 구조화 프로젝트

핵심 모듈:
- detection: YOLO 기반 레이아웃 감지
- ocr: EasyOCR 기반 텍스트 추출
- parser: 표 구조 파싱 및 JSON 변환
- utils: 유틸리티 함수
"""

__version__ = "0.1.0"

from .detection import LayoutDetector
from .ocr import OCRProcessor
from .parser import TableParser, DocumentParser
from .utils import get_device

__all__ = [
    "LayoutDetector",
    "OCRProcessor",
    "TableParser",
    "DocumentParser",
    "get_device",
]
