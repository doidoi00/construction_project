"""
DocLayNet 데이터셋 다운로드 스크립트

Hugging Face에서 DocLayNet 데이터셋을 다운로드하고 COCO 형식으로 변환합니다.
- 69,375 train 샘플
- 6,489 validation 샘플
- 4,999 test 샘플
- 11개 레이아웃 클래스

사용법:
    python src/data_preparation/download_doclaynet.py --output data/doclaynet

주의:
    datasets==2.14.5 버전 필요 (loading script 지원)
"""

import argparse
import json
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm


def download_doclaynet(output_dir: str):
    """
    DocLayNet 데이터셋을 Hugging Face에서 다운로드합니다.

    Args:
        output_dir: 다운로드할 디렉토리 경로

    DocLayNet 클래스 (11개):
        0: Caption - 이미지/표 캡션
        1: Footnote - 각주
        2: Formula - 수식
        3: List-item - 리스트 항목
        4: Page-footer - 페이지 하단
        5: Page-header - 페이지 상단
        6: Picture - 이미지/그림
        7: Section-header - 섹션 헤더
        8: Table - 표
        9: Text - 본문 텍스트
        10: Title - 제목

    데이터 구조:
        - image_id: 이미지 고유 ID
        - image: PIL 이미지 객체
        - width, height: 이미지 크기
        - doc_category: 문서 카테고리
        - objects: 바운딩 박스 어노테이션
            - bbox: [x_min, y_min, width, height]
            - category_id: 클래스 ID (0-10)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("DocLayNet 데이터셋 다운로드")
    print("=" * 70)
    print(f"저장 경로: {output_path}")
    print("=" * 70 + "\n")

    try:
        print("Hugging Face에서 데이터셋 로딩 중...")

        # Hugging Face에서 데이터셋 로드 (datasets==2.14.5 필요)
        dataset = load_dataset("ds4sd/DocLayNet")

        print(f"\n데이터셋 정보:")
        print(f"  - Train: {len(dataset['train']):,} 샘플")
        print(f"  - Validation: {len(dataset['validation']):,} 샘플")
        print(f"  - Test: {len(dataset['test']):,} 샘플")
        print(f"  - 총 {len(dataset['train']) + len(dataset['validation']) + len(dataset['test']):,} 샘플")

        # 각 분할(split)을 저장
        for split_name in ['train', 'validation', 'test']:
            split_dir = output_path / split_name
            split_dir.mkdir(parents=True, exist_ok=True)

            images_dir = split_dir / "images"
            images_dir.mkdir(exist_ok=True)

            print(f"\n{'='*70}")
            print(f"{split_name.upper()} 데이터 저장 중...")
            print(f"{'='*70}")

            split_data = dataset[split_name]

            # COCO 형식 어노테이션 준비
            coco_annotations = {
                "images": [],
                "annotations": [],
                "categories": [
                    {"id": 0, "name": "Caption"},
                    {"id": 1, "name": "Footnote"},
                    {"id": 2, "name": "Formula"},
                    {"id": 3, "name": "List-item"},
                    {"id": 4, "name": "Page-footer"},
                    {"id": 5, "name": "Page-header"},
                    {"id": 6, "name": "Picture"},
                    {"id": 7, "name": "Section-header"},
                    {"id": 8, "name": "Table"},
                    {"id": 9, "name": "Text"},
                    {"id": 10, "name": "Title"}
                ]
            }

            ann_id = 0

            # 각 샘플 처리
            for idx, sample in enumerate(tqdm(split_data, desc=f"Processing {split_name}")):
                # 이미지 저장
                image = sample['image']
                image_filename = f"{sample['image_id']}.png"
                image_path = images_dir / image_filename

                # PIL 이미지를 PNG로 저장
                image.save(image_path)

                # COCO 이미지 정보
                coco_annotations["images"].append({
                    "id": idx,
                    "file_name": image_filename,
                    "width": sample['width'],
                    "height": sample['height'],
                    "doc_category": sample.get('doc_category', ''),
                    "image_id": sample['image_id']
                })

                # COCO 어노테이션 정보
                # objects는 여러 개의 바운딩 박스를 포함
                if 'objects' in sample and sample['objects'] is not None:
                    objects = sample['objects']
                    
                    # dict 형태: {'bbox': [...], 'category_id': [...]}
                    if isinstance(objects, dict):
                        bboxes = objects.get('bbox', [])
                        category_ids = objects.get('category_id', [])
                        
                        for bbox, category_id in zip(bboxes, category_ids):
                            # bbox: [x_min, y_min, width, height] (COCO 형식)
                            coco_annotations["annotations"].append({
                                "id": ann_id,
                                "image_id": idx,
                                "category_id": category_id,
                                "bbox": bbox,
                                "area": bbox[2] * bbox[3],  # width * height
                                "iscrowd": 0
                            })
                            ann_id += 1
                    
                    # list 형태: [{'bbox': [...], 'category_id': ...}, ...]
                    elif isinstance(objects, list):
                        for obj in objects:
                            bbox = obj.get('bbox', obj.get('bboxes', []))
                            category_id = obj.get('category_id', obj.get('category', 0))
                            
                            if bbox:
                                coco_annotations["annotations"].append({
                                    "id": ann_id,
                                    "image_id": idx,
                                    "category_id": category_id,
                                    "bbox": bbox,
                                    "area": bbox[2] * bbox[3],  # width * height
                                    "iscrowd": 0
                                })
                                ann_id += 1

            # COCO JSON 저장
            json_path = split_dir / "annotations.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(coco_annotations, f, indent=2)

            print(f"\n✅ {split_name} 저장 완료:")
            print(f"   - 이미지: {len(coco_annotations['images']):,}개")
            print(f"   - 어노테이션: {ann_id:,}개")
            print(f"   - JSON: {json_path}")

        print("\n" + "=" * 70)
        print("✅ DocLayNet 다운로드 완료!")
        print("=" * 70)

        # 데이터셋 구조 안내
        print("\n📁 데이터셋 구조:")
        print(f"{output_path}/")
        print("  ├── train/")
        print("  │   ├── images/")
        print("  │   │   └── *.png")
        print("  │   └── annotations.json")
        print("  ├── validation/")
        print("  │   ├── images/")
        print("  │   │   └── *.png")
        print("  │   └── annotations.json")
        print("  └── test/")
        print("      ├── images/")
        print("      │   └── *.png")
        print("      └── annotations.json")

        print("\n다음 단계: COCO → YOLO 형식 변환")
        print(f"  python src/data_preparation/convert_to_yolo.py \\")
        print(f"    --input {output_path} \\")
        print(f"    --output {output_path / 'yolo'}")

    except Exception as e:
        print(f"\n❌ 에러 발생: {e}")
        print("\n문제 해결:")
        print("1. datasets 버전 확인:")
        print("   uv pip install 'datasets==2.14.5'")
        print("2. 인터넷 연결 확인")
        print("3. 충분한 디스크 공간 확인 (약 30GB 필요)")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="DocLayNet 데이터셋 다운로드"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/doclaynet",
        help="다운로드할 디렉토리 경로 (기본값: data/doclaynet)"
    )

    args = parser.parse_args()

    print("\n📊 DocLayNet 데이터셋 정보:")
    print("  - 출처: IBM Research")
    print("  - 총 샘플: ~80,000 페이지")
    print("  - 클래스: 11개 레이아웃 요소")
    print("  - 이미지 크기: 1025x1025 px")
    print("  - 형식: PNG + COCO JSON")
    print("\n⚠️ 처음 다운로드 시 ~30GB의 데이터가 다운로드됩니다.")
    print("⚠️ datasets==2.14.5 버전 필요 (loading script 지원)")

    download_doclaynet(args.output)


if __name__ == "__main__":
    main()
