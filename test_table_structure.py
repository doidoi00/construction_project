"""
Table Transformer 기능 테스트 스크립트

이 스크립트는 Table Transformer의 기본 기능을 테스트합니다.
실제 이미지 파일이 있을 때 사용하세요.
"""

import sys
from pathlib import Path

# 테스트용 이미지 경로 (사용자가 수정 필요)
TEST_IMAGE = "data/test_table.png"  # 테스트할 표 이미지 경로


def test_table_transformer():
    """
    Table Transformer의 기본 기능을 테스트합니다.
    """
    from src.table_structure import TableStructureRecognizer
    from src.utils import get_device

    print("\n" + "="*60)
    print("Table Transformer 기능 테스트")
    print("="*60)

    # 테스트 이미지 확인
    if not Path(TEST_IMAGE).exists():
        print(f"\n❌ 테스트 이미지를 찾을 수 없습니다: {TEST_IMAGE}")
        print("\n테스트를 진행하려면:")
        print("1. PDF에서 표가 포함된 페이지를 이미지로 추출")
        print("2. 이미지를 data/test_table.png로 저장")
        print("3. 이 스크립트를 다시 실행")
        return

    # 디바이스 설정
    device = get_device()
    print(f"\n실행 디바이스: {device}")

    # Table Transformer 초기화
    print("\nTable Transformer 모델 로딩 중...")
    recognizer = TableStructureRecognizer(
        device=device,
        confidence_threshold=0.7
    )

    # 표 구조 인식
    print(f"\n표 구조 인식 중: {TEST_IMAGE}")
    structure = recognizer.recognize_structure(TEST_IMAGE)

    # 결과 출력
    print("\n" + "="*60)
    print("구조 인식 결과")
    print("="*60)

    print(f"\n이미지 크기: {structure['image_size']}")
    print(f"감지된 행: {len(structure['rows'])}개")
    print(f"감지된 열: {len(structure['columns'])}개")
    print(f"병합 셀: {len(structure['spanning_cells'])}개")

    # 행 정보 출력
    if structure['rows']:
        print("\n[행 정보]")
        for i, row in enumerate(structure['rows']):
            print(f"  Row {i+1}: bbox={row['bbox']}, confidence={row['confidence']:.2f}")

    # 열 정보 출력
    if structure['columns']:
        print("\n[열 정보]")
        for i, col in enumerate(structure['columns']):
            print(f"  Col {i+1}: bbox={col['bbox']}, confidence={col['confidence']:.2f}")

    # 셀 그리드 생성
    print("\n셀 그리드 생성 중...")
    cell_grid = recognizer.create_cell_grid(structure)

    if cell_grid:
        print(f"✓ 셀 그리드 생성 완료: {len(cell_grid)}x{len(cell_grid[0])}")

        # 병합 셀 정보
        merged_count = 0
        for row in cell_grid:
            for cell in row:
                if cell and cell['is_spanning'] and (cell['row_span'] > 1 or cell['col_span'] > 1):
                    merged_count += 1

        if merged_count > 0:
            print(f"  └─ 병합 셀: {merged_count}개")

    print("\n" + "="*60)
    print("✓ 테스트 완료!")
    print("="*60 + "\n")


def test_full_pipeline():
    """
    YOLO + Table Transformer + OCR 전체 파이프라인을 테스트합니다.
    """
    print("\n" + "="*60)
    print("전체 파이프라인 테스트")
    print("="*60)
    print("\n전체 파이프라인을 테스트하려면 다음 명령어를 실행하세요:")
    print("\npython main.py <이미지_경로> -o data/output\n")
    print("예시:")
    print("  python main.py data/test_table.png")
    print("  python main.py data/images/  # 디렉토리 일괄 처리")
    print("\n옵션:")
    print("  --no-structure    : Table Transformer 사용 안 함 (기존 방식)")
    print("  --conf 0.5        : YOLO 신뢰도 임계값")
    print("  --structure-conf 0.7 : Table Transformer 신뢰도 임계값")
    print("="*60 + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Table Transformer 기능 테스트")
    parser.add_argument(
        "--full",
        action="store_true",
        help="전체 파이프라인 테스트 정보 출력"
    )

    args = parser.parse_args()

    if args.full:
        test_full_pipeline()
    else:
        test_table_transformer()
