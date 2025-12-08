"""
건설 시방서 표 구조화 - 메인 파이프라인

전체 처리 흐름:
1. PDF를 이미지로 변환 (선택적)
2. YOLO로 문서 레이아웃 감지 (표 영역 탐지)
3. Table Transformer로 표 구조 인식 (행/열/셀 감지)
4. OCR로 감지된 표 영역에서 텍스트 추출
5. 텍스트를 셀 그리드에 매핑
6. 구조화된 JSON으로 저장
"""
import os
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ['NCCL_IB_DISABLE'] = '1'

import argparse
from pathlib import Path
from typing import List, Dict
import cv2
import numpy as np

from src.detection import LayoutDetector
from src.table_structure import (
    TableStructureRecognizer,
    match_ocr_to_cells,
    create_structured_table
)
from src.ocr import OCRProcessor
from src.parser import DocumentParser, validate_table_structure
from src.utils import get_device


def process_single_image(
    image_path: str,
    detector: LayoutDetector,
    structure_recognizer: TableStructureRecognizer,
    ocr_processor: OCRProcessor,
    output_dir: str,
    use_structure_recognition: bool = True,
    save_cropped_tables: bool = True
) -> tuple[List[Dict], str]:
    """
    단일 이미지를 처리하여 표를 추출합니다.

    Args:
        image_path: 입력 이미지 경로
        detector: YOLO 감지기
        structure_recognizer: Table Transformer 구조 인식기
        ocr_processor: OCR 처리기
        output_dir: 출력 디렉토리 (기본 output 디렉토리)
        use_structure_recognition: Table Transformer 사용 여부
        save_cropped_tables: 크롭된 표 이미지 저장 여부

    Returns:
        tuple[List[Dict], str]: (추출된 표 리스트, 페이지 출력 디렉토리)

    처리 단계:
        1. YOLO로 표 영역 감지
        2. Table Transformer로 표 구조 인식 (행/열/셀)
        3. OCR로 텍스트 추출
        4. 텍스트를 셀 그리드에 매핑
        5. 구조화된 표 데이터 생성
        6. 크롭된 표 이미지 저장
    """
    # 이미지 경로에서 PDF 이름과 페이지 번호 추출
    # 예: data/images/spec_001/page_001.png -> spec_001, page_001
    image_path_obj = Path(image_path)
    page_name = image_path_obj.stem  # page_001
    pdf_name = image_path_obj.parent.name  # spec_001

    print(f"\n{'='*60}")
    print(f"처리 중: {pdf_name}/{page_name}")
    print(f"{'='*60}")

    # 1. 표 영역 감지
    print("\n[1/5] YOLO 표 영역 감지 중...")
    detections = detector.detect_tables(image_path)

    if not detections:
        print("⚠ 표가 감지되지 않았습니다. 폴더를 생성하지 않습니다.")
        return [], None

    print(f"✓ {len(detections)}개 표 감지됨")

    # 표가 감지되었으므로 폴더 생성 (pdf_name/page_name 구조)
    page_output_dir = Path(output_dir) / pdf_name / page_name
    page_output_dir.mkdir(parents=True, exist_ok=True)

    # 원본 이미지 로드
    original_image = cv2.imread(image_path)
    tables = []

    for idx, detection in enumerate(detections):
        print(f"\n{'='*60}")
        print(f"표 {idx + 1}/{len(detections)} 처리 중...")
        print(f"{'='*60}")

        bbox = detection["bbox"]
        confidence = detection["confidence"]

        # 표 영역 크롭 (아래쪽에 여유 마진 추가)
        x1, y1, x2, y2 = map(int, bbox)
        img_height, img_width = original_image.shape[:2]
        bottom_margin = 5  # 아래쪽 마진 (픽셀)
        y2 = min(y2 + bottom_margin, img_height)  # 이미지 경계를 넘지 않도록
        table_image = original_image[y1:y2, x1:x2]

        # 크롭된 표 이미지 저장
        table_image_filename = f"table_{idx + 1:02d}.png"
        table_image_path = page_output_dir / table_image_filename
        cv2.imwrite(str(table_image_path), table_image)
        print(f"  ✓ 표 이미지 저장: {table_image_filename}")

        if use_structure_recognition:
            # 2. Table Transformer로 표 구조 인식
            print("\n[2/5] Table Transformer 구조 인식 중...")
            structure = structure_recognizer.recognize_structure(str(table_image_path))

            # 3. 셀 그리드 생성
            print("\n[3/5] 셀 그리드 생성 중...")
            cell_grid = structure_recognizer.create_cell_grid(structure)

            # 4. OCR 텍스트 추출
            print("\n[4/5] OCR 텍스트 추출 중...")
            ocr_results = ocr_processor.extract_text(str(table_image_path))

            if not ocr_results:
                print("  ⚠ 텍스트가 추출되지 않았습니다.")
                continue

            # OCR 결과를 셀 그리드에 맞게 변환
            ocr_data = []
            for result in ocr_results:
                # extract_text()는 딕셔너리 리스트를 반환
                text = result["text"]
                bbox_ocr = result["bbox"]

                # bbox 형식 변환: [[x1,y1], [x2,y1], [x2,y2], [x1,y2]] -> [x1, y1, x2, y2]
                x_coords = [point[0] for point in bbox_ocr]
                y_coords = [point[1] for point in bbox_ocr]
                ocr_bbox = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]

                ocr_data.append({
                    "text": text,
                    "bbox": ocr_bbox,
                    "confidence": result["confidence"]
                })

            # 5. 텍스트를 셀에 매핑
            print("\n[5/5] 텍스트를 셀에 매핑 중...")
            text_grid = match_ocr_to_cells(ocr_data, cell_grid)

            # 구조화된 표 생성
            table_content = create_structured_table(cell_grid, text_grid)

        else:
            # 기존 방식: OCR만 사용
            print("\n[2/3] OCR 텍스트 추출 중...")
            from src.ocr import cluster_texts_by_line

            # OCR 실행
            text_data = ocr_processor.extract_text_from_region(image_path, bbox)

            if not text_data:
                print("  ⚠ 텍스트가 추출되지 않았습니다.")
                continue

            # 텍스트를 행별로 그룹화
            print("\n[3/3] 표 구조 분석 중...")
            text_lines = cluster_texts_by_line(text_data, y_threshold=15.0)

            # 간단한 표 구조 생성
            table_data = []
            max_cols = 0

            for line in text_lines:
                row_texts = [item["text"] for item in line]
                table_data.append(row_texts)
                max_cols = max(max_cols, len(row_texts))

            # 열 수 맞추기
            for row in table_data:
                while len(row) < max_cols:
                    row.append("")

            table_content = {
                "rows": len(table_data),
                "cols": max_cols,
                "data": table_data,
                "merged_cells": []
            }

        # 표 구조 생성
        table_structure = {
            "type": "table",
            "bbox": bbox,
            "confidence": confidence,
            "content": table_content
        }

        # 검증
        if validate_table_structure(table_structure):
            # 표 이미지 파일 경로 추가
            table_structure["image_path"] = str(table_image_path.relative_to(Path(output_dir)))
            tables.append(table_structure)
            print(f"\n✓ 표 {idx + 1} 파싱 완료: "
                  f"{table_structure['content']['rows']}행 x "
                  f"{table_structure['content']['cols']}열")

            if table_structure['content']['merged_cells']:
                print(f"  └─ 병합 셀: {len(table_structure['content']['merged_cells'])}개")
        else:
            print(f"  ⚠ 표 {idx + 1} 검증 실패, 건너뜀")

    return tables, str(page_output_dir)


def main():
    """
    메인 함수 - 커맨드라인 인터페이스
    """
    # 커맨드라인 인자 파싱
    parser = argparse.ArgumentParser(
        description="건설 시방서 표 구조화 도구 (YOLO + Table Transformer + OCR)"
    )

    parser.add_argument(
        "input",
        type=str,
        help="입력 이미지 파일 또는 디렉토리 경로"
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        default="data/output",
        help="출력 디렉토리 (기본값: data/output)"
    )

    parser.add_argument(
        "-m", "--model",
        type=str,
        default="models/finetuned/weights/best.pt",
        help="YOLO 모델 경로 (기본값: models/finetuned/weights/best.pt)"
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="YOLO 감지 신뢰도 임계값 (기본값: 0.5)"
    )

    parser.add_argument(
        "--structure-conf",
        type=float,
        default=0.7,
        help="Table Transformer 신뢰도 임계값 (기본값: 0.7)"
    )

    parser.add_argument(
        "--lang",
        type=str,
        default="en",
        help="OCR 언어 (기본값: ko,en)"
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="디바이스 (mps/cpu, 기본값: 자동 선택)"
    )

    parser.add_argument(
        "--no-structure",
        action="store_true",
        help="Table Transformer 사용 안 함 (기존 방식)"
    )

    args = parser.parse_args()

    # 입력 경로 확인
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ 입력 경로를 찾을 수 없습니다: {args.input}")
        return

    # 출력 디렉토리 생성
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("건설 시방서 표 구조화 도구")
    print("YOLO + Table Transformer + OCR 파이프라인")
    print("="*60)

    # 디바이스 설정
    device = args.device if args.device else get_device()
    print(f"\n실행 디바이스: {device}")

    # 모듈 초기화
    print("\n모듈 초기화 중...")

    # 1. YOLO 레이아웃 감지기
    detector = LayoutDetector(
        model_path=args.model,
        device=device,
        confidence_threshold=args.conf
    )

    # 2. Table Transformer 구조 인식기 (옵션)
    structure_recognizer = None
    if not args.no_structure:
        structure_recognizer = TableStructureRecognizer(
            device=device,
            confidence_threshold=args.structure_conf
        )

    # 3. OCR 처리기
    languages = args.lang.split(',')
    use_gpu = device != 'cpu'
    # MPS의 경우 EasyOCR 이슈가 있을 수 있으나, 사용자 요청에 따라 활성화 시도
    ocr_processor = OCRProcessor(languages=languages, gpu=use_gpu)

    # 4. 문서 파서
    doc_parser = DocumentParser()

    # 입력이 디렉토리인 경우
    if input_path.is_dir():
        image_files = []
        # 재귀적으로 모든 하위 디렉토리에서 이미지 파일 찾기
        for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
            image_files.extend(input_path.rglob(f"*{ext}"))

        if not image_files:
            print(f"⚠ {input_path}에 이미지 파일이 없습니다.")
            return

        print(f"\n{len(image_files)}개 이미지 파일 발견 (하위 폴더 포함)")

        # 배치 처리
        for image_file in sorted(image_files):
            tables, page_dir = process_single_image(
                str(image_file),
                detector,
                structure_recognizer,
                ocr_processor,
                str(output_dir),
                use_structure_recognition=not args.no_structure
            )

            # JSON 저장 (페이지별 디렉토리에 저장)
            if tables:
                # 페이지 이름으로 document ID 생성
                page_name = image_file.stem
                document = doc_parser.create_document_structure(
                    page_name,
                    tables
                )

                # JSON을 페이지 디렉토리에 저장
                json_path = Path(page_dir) / f"{page_name}.json"
                doc_parser.save_to_json(document, str(json_path))
                print(f"  ✓ JSON 저장: {json_path.relative_to(output_dir)}")

    # 입력이 단일 파일인 경우
    else:
        tables, page_dir = process_single_image(
            str(input_path),
            detector,
            structure_recognizer,
            ocr_processor,
            str(output_dir),
            use_structure_recognition=not args.no_structure
        )

        # JSON 저장 (페이지별 디렉토리에 저장)
        if tables:
            page_name = input_path.stem
            document = doc_parser.create_document_structure(
                page_name,
                tables
            )

            # JSON을 페이지 디렉토리에 저장
            json_path = Path(page_dir) / f"{page_name}.json"
            doc_parser.save_to_json(document, str(json_path))
            print(f"  ✓ JSON 저장: {json_path.relative_to(output_dir)}")

    print("\n" + "="*60)
    print("✓ 처리 완료!")
    print(f"결과 저장 위치: {output_dir}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
