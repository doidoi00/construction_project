"""
COCO 형식 → YOLO 형식 변환 스크립트

DocLayNet의 COCO JSON 어노테이션을 YOLO 텍스트 형식으로 변환합니다.

COCO 형식:
    - JSON 파일에 모든 어노테이션 저장
    - bbox: [x_min, y_min, width, height] (픽셀 단위)

YOLO 형식:
    - 이미지당 하나의 txt 파일
    - 각 줄: <class_id> <x_center> <y_center> <width> <height>
    - 좌표는 이미지 크기로 정규화 (0~1)

사용법:
    python src/data_preparation/convert_to_yolo.py \
        --input data/doclaynet \
        --output data/doclaynet/yolo
"""

import argparse
import json
import shutil
from pathlib import Path
from tqdm import tqdm


def coco_to_yolo_bbox(coco_bbox, img_width, img_height):
    """
    COCO 바운딩 박스를 YOLO 형식으로 변환합니다.

    Args:
        coco_bbox: [x_min, y_min, width, height] (픽셀 단위)
        img_width: 이미지 너비
        img_height: 이미지 높이

    Returns:
        [x_center, y_center, width, height] (정규화된 값, 0~1)

    변환 공식:
        x_center = (x_min + width/2) / img_width
        y_center = (y_min + height/2) / img_height
        norm_width = width / img_width
        norm_height = height / img_height
    """
    x_min, y_min, bbox_width, bbox_height = coco_bbox

    # 중심 좌표 계산
    x_center = (x_min + bbox_width / 2) / img_width
    y_center = (y_min + bbox_height / 2) / img_height

    # 너비/높이 정규화
    norm_width = bbox_width / img_width
    norm_height = bbox_height / img_height

    # YOLO는 0~1 범위로 클리핑
    x_center = max(0, min(1, x_center))
    y_center = max(0, min(1, y_center))
    norm_width = max(0, min(1, norm_width))
    norm_height = max(0, min(1, norm_height))

    return [x_center, y_center, norm_width, norm_height]


def convert_split(input_dir: Path, output_dir: Path, split_name: str):
    """
    하나의 데이터 분할(train/val/test)을 변환합니다.

    Args:
        input_dir: COCO 형식 데이터 경로
        output_dir: YOLO 형식 데이터 저장 경로
        split_name: 'train', 'validation', 'test'

    처리 과정:
        1. COCO JSON 파일 로드
        2. 각 이미지에 대해:
           - 이미지 파일 복사
           - 해당 이미지의 모든 어노테이션을 YOLO 형식으로 변환
           - txt 파일로 저장
    """
    split_input = input_dir / split_name
    json_path = split_input / "annotations.json"

    # YOLO 디렉토리 구조
    # validation → val로 이름 변경
    yolo_split_name = "val" if split_name == "validation" else split_name

    images_output = output_dir / "images" / yolo_split_name
    labels_output = output_dir / "labels" / yolo_split_name

    images_output.mkdir(parents=True, exist_ok=True)
    labels_output.mkdir(parents=True, exist_ok=True)

    print(f"\n{split_name} 데이터 변환 중...")

    # COCO JSON 로드
    with open(json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)

    images = coco_data['images']
    annotations = coco_data['annotations']

    # 이미지 ID별로 어노테이션 그룹화
    # {image_id: [annotation1, annotation2, ...]}
    img_to_anns = {}
    for ann in annotations:
        img_id = ann['image_id']
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(ann)

    # 각 이미지 처리
    for img_info in tqdm(images, desc=f"Converting {split_name}"):
        img_id = img_info['id']
        img_filename = img_info['file_name']
        img_width = img_info['width']
        img_height = img_info['height']

        # 이미지 파일 복사
        src_image = split_input / "images" / img_filename
        dst_image = images_output / img_filename

        if src_image.exists():
            shutil.copy2(src_image, dst_image)
        else:
            print(f"  ⚠️ 이미지 파일 없음: {src_image}")
            continue

        # YOLO 라벨 파일 생성
        label_filename = Path(img_filename).stem + ".txt"
        label_path = labels_output / label_filename

        # 해당 이미지의 어노테이션 변환
        yolo_lines = []
        if img_id in img_to_anns:
            for ann in img_to_anns[img_id]:
                class_id = ann['category_id']
                coco_bbox = ann['bbox']

                # COCO → YOLO 변환
                yolo_bbox = coco_to_yolo_bbox(coco_bbox, img_width, img_height)

                # YOLO 형식: <class_id> <x_center> <y_center> <width> <height>
                yolo_line = f"{class_id} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}"
                yolo_lines.append(yolo_line)

        # 라벨 파일 저장 (어노테이션이 없어도 빈 파일 생성)
        with open(label_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(yolo_lines))

    print(f"✅ {split_name} 변환 완료: {len(images)} 이미지")


def create_yaml(output_dir: Path, project_root: Path):
    """
    YOLO 데이터셋 설정 파일 (doclaynet.yaml) 생성

    이 파일은 YOLO 학습 시 데이터셋 경로와 클래스 정보를 제공합니다.

    Args:
        output_dir: YOLO 데이터셋 경로
        project_root: 프로젝트 루트 경로
    """
    # output_dir을 절대 경로로 변환한 후 상대 경로 계산
    # 이렇게 하면 상대 경로로 전달된 경우에도 정상 동작
    output_dir_abs = output_dir.resolve()
    project_root_abs = project_root.resolve()
    data_path = output_dir_abs.relative_to(project_root_abs)

    yaml_content = f"""# DocLayNet 데이터셋 설정
# 문서 레이아웃 분석용 데이터셋

# 데이터셋 경로 (프로젝트 루트 기준 상대 경로)
path: {data_path}
train: images/train
val: images/val
test: images/test

# 클래스 정의 (11개)
names:
  0: Caption        # 이미지/표 캡션
  1: Footnote       # 각주
  2: Formula        # 수식
  3: List-item      # 리스트 항목
  4: Page-footer    # 페이지 하단
  5: Page-header    # 페이지 상단
  6: Picture        # 이미지/그림
  7: Section-header # 섹션 헤더
  8: Table          # 표
  9: Text           # 본문 텍스트
  10: Title         # 제목

# 클래스 개수
nc: 11
"""

    yaml_path = output_dir / "doclaynet.yaml"
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)

    print(f"\n✅ 설정 파일 생성: {yaml_path}")
    print(f"\n다음 단계: 모델 학습")
    print(f"  python src/training/train_doclaynet.py --data {yaml_path}")


def main():
    parser = argparse.ArgumentParser(
        description="COCO 형식을 YOLO 형식으로 변환"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/doclaynet",
        help="COCO 형식 데이터셋 경로 (기본값: data/doclaynet)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/doclaynet/yolo",
        help="YOLO 형식 데이터셋 저장 경로 (기본값: data/doclaynet/yolo)"
    )

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    project_root = Path.cwd()

    print("=" * 70)
    print("COCO → YOLO 형식 변환")
    print("=" * 70)
    print(f"입력 경로: {input_dir}")
    print(f"출력 경로: {output_dir}")

    # 출력 디렉토리 생성
    output_dir.mkdir(parents=True, exist_ok=True)

    # 각 분할 변환
    for split in ['train', 'validation', 'test']:
        split_path = input_dir / split
        if split_path.exists():
            convert_split(input_dir, output_dir, split)
        else:
            print(f"\n⚠️ {split} 데이터 없음: {split_path}")

    # YAML 설정 파일 생성
    create_yaml(output_dir, project_root)

    print("\n" + "=" * 70)
    print("✅ 변환 완료!")
    print("=" * 70)


if __name__ == "__main__":
    main()
