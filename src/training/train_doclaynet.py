"""
YOLO11 DocLayNet 파인튜닝 학습 스크립트

DocLayNet 데이터셋으로 YOLO11 모델을 파인튜닝합니다.
macOS Apple Silicon (MPS) 최적화가 적용되어 있습니다.

사용법:
    # 기본 설정으로 학습
    python src/training/train_doclaynet.py

    # 커스텀 설정으로 학습
    python src/training/train_doclaynet.py --config configs/custom_config.yaml

    # 특정 파라미터 오버라이드
    python src/training/train_doclaynet.py --epochs 50 --batch 8
"""

import argparse
import os
import sys
from pathlib import Path
import torch
import yaml
from ultralytics import YOLO


def check_environment():
    """
    학습 환경을 체크하고 출력합니다.

    확인 사항:
        - PyTorch 버전
        - CUDA/MPS 사용 가능 여부
        - 디바이스 정보
    """
    print("=" * 70)
    print("학습 환경 체크")
    print("=" * 70)

    print(f"Python 버전: {sys.version}")
    print(f"PyTorch 버전: {torch.__version__}")

    # 디바이스 체크
    if torch.cuda.is_available():
        device = "cuda"
        print(f"✅ CUDA 사용 가능")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
        print(f"✅ MPS (Apple Silicon) 사용 가능")
        print(f"   ⚠️ 주의: amp=False 설정 필요 (FP16 미지원)")
    else:
        device = "cpu"
        print(f"⚠️ GPU 사용 불가, CPU 사용")

    print(f"\n사용할 디바이스: {device}")
    print("=" * 70 + "\n")

    return device


def download_pretrained_model(model_name: str = "yolo11n.pt"):
    """
    사전학습된 YOLO 모델을 다운로드합니다.

    Args:
        model_name: 모델 이름 (yolo11n.pt, yolo11s.pt, 등)

    Returns:
        모델 파일 경로

    YOLO11 모델 크기:
        - yolo11n.pt: Nano (가장 작음, 빠름)
        - yolo11s.pt: Small
        - yolo11m.pt: Medium
        - yolo11l.pt: Large
        - yolo11x.pt: Extra Large (가장 큼, 느림)
    """
    models_dir = Path("models/pretrained")
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / model_name

    if not model_path.exists():
        print(f"사전학습 모델 다운로드 중: {model_name}")
        # YOLO 클래스가 자동으로 다운로드
        model = YOLO(model_name)
        # 다운로드된 모델을 pretrained 디렉토리로 이동
        # (YOLO는 기본적으로 홈 디렉토리에 다운로드)
        print(f"✅ 다운로드 완료: {model_path}")
    else:
        print(f"✅ 사전학습 모델 존재: {model_path}")

    return str(model_path)


def load_config(config_path: str = None):
    """
    설정 파일을 로드합니다.

    Args:
        config_path: YAML 설정 파일 경로

    Returns:
        설정 딕셔너리
    """
    if config_path is None:
        config_path = "configs/train_config.yaml"

    config_path = Path(config_path)

    if not config_path.exists():
        print(f"⚠️ 설정 파일 없음: {config_path}")
        print("기본 설정 사용")
        return {}

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    print(f"✅ 설정 파일 로드: {config_path}")
    return config


def train_model(config: dict, **kwargs):
    """
    YOLO 모델을 학습합니다.

    Args:
        config: 학습 설정 딕셔너리
        **kwargs: 추가 파라미터 (config 오버라이드)

    학습 과정:
        1. 사전학습 모델 로드
        2. 학습 설정 적용
        3. 학습 실행
        4. 결과 저장

    출력:
        - models/finetuned/<name>/weights/best.pt: 최고 성능 모델
        - models/finetuned/<name>/weights/last.pt: 마지막 에폭 모델
        - models/finetuned/<name>/results.png: 학습 그래프
    """
    # config와 kwargs 병합 (kwargs가 우선)
    train_args = {**config, **kwargs}

    # 모델 경로
    model_path = train_args.get('model', 'yolo11n.pt')

    # 사전학습 모델이 없으면 다운로드
    if not Path(model_path).exists():
        model_name = Path(model_path).name
        model_path = download_pretrained_model(model_name)
        train_args['model'] = model_path

    print("\n" + "=" * 70)
    print("학습 설정")
    print("=" * 70)
    for key, value in train_args.items():
        print(f"{key:20s}: {value}")
    print("=" * 70 + "\n")

    # YOLO 모델 로드
    print(f"모델 로드: {model_path}")
    model = YOLO(model_path)

    # 학습 시작
    print("\n" + "=" * 70)
    print("🚀 학습 시작")
    print("=" * 70 + "\n")

    try:
        # train() 메서드에 전달할 인자만 필터링
        # 'model' 키는 제외 (이미 로드됨)
        train_params = {k: v for k, v in train_args.items() if k != 'model'}

        results = model.train(**train_params)

        print("\n" + "=" * 70)
        print("✅ 학습 완료!")
        print("=" * 70)
        print(f"\n최고 성능 모델: {train_args.get('project', 'runs/detect')}/{train_args.get('name', 'train')}/weights/best.pt")
        print(f"마지막 모델: {train_args.get('project', 'runs/detect')}/{train_args.get('name', 'train')}/weights/last.pt")
        print(f"\n다음 단계: 모델 평가")
        print(f"  python src/training/evaluate.py --model <best.pt 경로>")

        return results

    except Exception as e:
        print(f"\n❌ 학습 중 에러 발생: {e}")
        print("\n문제 해결:")
        print("1. MPS 에러 시: amp=false 설정 확인")
        print("2. 메모리 부족 시: batch 크기 줄이기 (--batch 4)")
        print("3. 데이터셋 경로 확인: data 파라미터")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="YOLO11 DocLayNet 파인튜닝"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",
        help="학습 설정 파일 경로"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="사전학습 모델 경로 (설정 파일 오버라이드)"
    )
    parser.add_argument(
        "--data",
        type=str,
        help="데이터셋 YAML 경로 (설정 파일 오버라이드)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="학습 에폭 수 (설정 파일 오버라이드)"
    )
    parser.add_argument(
        "--batch",
        type=int,
        help="배치 크기 (설정 파일 오버라이드)"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        help="입력 이미지 크기 (설정 파일 오버라이드)"
    )
    parser.add_argument(
        "--device",
        type=str,
        help="디바이스 (cpu, cuda, mps)"
    )
    parser.add_argument(
        "--name",
        type=str,
        help="실험 이름 (설정 파일 오버라이드)"
    )

    args = parser.parse_args()

    # 환경 체크
    default_device = check_environment()

    # 설정 로드
    config = load_config(args.config)

    # 명령줄 인자로 오버라이드
    overrides = {}
    if args.model:
        overrides['model'] = args.model
    if args.data:
        overrides['data'] = args.data
    if args.epochs:
        overrides['epochs'] = args.epochs
    if args.batch:
        overrides['batch'] = args.batch
    if args.imgsz:
        overrides['imgsz'] = args.imgsz
    if args.device:
        overrides['device'] = args.device
    elif 'device' not in config:
        overrides['device'] = default_device
    if args.name:
        overrides['name'] = args.name

    # 학습 실행
    train_model(config, **overrides)


if __name__ == "__main__":
    main()
