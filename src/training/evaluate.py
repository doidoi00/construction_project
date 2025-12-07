"""
학습된 YOLO 모델 평가 스크립트

학습된 모델의 성능을 테스트 데이터셋으로 평가하고,
클래스별 상세 성능 분석 및 시각화를 제공합니다.

사용법:
    # 기본 평가
    python src/training/evaluate.py \
        --model models/finetuned/doclaynet_yolo11n/weights/best.pt \
        --data data/doclaynet/yolo/doclaynet.yaml

    # 특정 분할 평가
    python src/training/evaluate.py \
        --model models/finetuned/best.pt \
        --data data/doclaynet/yolo/doclaynet.yaml \
        --split test
"""

import argparse
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO


def evaluate_model(model_path: str, data_path: str, split: str = "val", save_dir: str = None):
    """
    모델 성능을 평가합니다.

    Args:
        model_path: 학습된 모델 경로 (.pt 파일)
        data_path: 데이터셋 YAML 파일 경로
        split: 평가할 데이터 분할 (val, test)
        save_dir: 결과 저장 디렉토리

    평가 지표:
        - mAP@0.5: IoU 0.5에서의 mean Average Precision
        - mAP@0.5:0.95: IoU 0.5~0.95에서의 mAP
        - Precision: 정밀도
        - Recall: 재현율
        - F1 Score: F1 점수

    출력:
        - 전체 성능 지표
        - 클래스별 성능 분석
        - 혼동 행렬 (Confusion Matrix)
        - 예측 결과 시각화
    """
    print("=" * 70)
    print("모델 평가")
    print("=" * 70)
    print(f"모델: {model_path}")
    print(f"데이터: {data_path}")
    print(f"분할: {split}")
    print("=" * 70 + "\n")

    # 모델 로드
    model = YOLO(model_path)

    # 평가 실행
    print("평가 시작...\n")
    results = model.val(
        data=data_path,
        split=split,
        save_json=True,  # COCO JSON 형식으로 결과 저장
        save_hybrid=True,  # 라벨과 예측 함께 저장
        plots=True  # 시각화 생성
    )

    # 결과 출력
    print("\n" + "=" * 70)
    print("📊 평가 결과")
    print("=" * 70)

    # 전체 성능 지표
    print("\n전체 성능:")
    print(f"  mAP@0.5:      {results.box.map50:.4f}")
    print(f"  mAP@0.5:0.95: {results.box.map:.4f}")
    print(f"  Precision:    {results.box.mp:.4f}")
    print(f"  Recall:       {results.box.mr:.4f}")

    # 클래스별 성능 (상위 3개 클래스)
    print("\n클래스별 성능 (mAP@0.5):")

    # DocLayNet 클래스 이름
    class_names = [
        "Caption", "Footnote", "Formula", "List-item",
        "Page-footer", "Page-header", "Picture", "Section-header",
        "Table", "Text", "Title"
    ]

    # 클래스별 AP 출력
    if hasattr(results.box, 'ap_class_index') and hasattr(results.box, 'ap50'):
        class_ap = {}
        for idx, ap in zip(results.box.ap_class_index, results.box.ap50):
            class_name = class_names[idx] if idx < len(class_names) else f"Class {idx}"
            class_ap[class_name] = ap
            print(f"  {class_name:15s}: {ap:.4f}")

        # Table 클래스 특화 분석
        if "Table" in class_ap:
            print(f"\n⭐ Table 클래스 성능: {class_ap['Table']:.4f}")
            if class_ap['Table'] > 0.9:
                print("   ✅ 우수한 표 감지 성능!")
            elif class_ap['Table'] > 0.7:
                print("   ✅ 양호한 표 감지 성능")
            else:
                print("   ⚠️ 표 감지 성능 개선 필요")

    print("\n" + "=" * 70)

    # 결과 저장
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # JSON으로 결과 저장
        results_dict = {
            "model": str(model_path),
            "data": str(data_path),
            "split": split,
            "metrics": {
                "mAP@0.5": float(results.box.map50),
                "mAP@0.5:0.95": float(results.box.map),
                "precision": float(results.box.mp),
                "recall": float(results.box.mr)
            }
        }

        if hasattr(results.box, 'ap_class_index') and hasattr(results.box, 'ap50'):
            results_dict["class_metrics"] = {}
            for idx, ap in zip(results.box.ap_class_index, results.box.ap50):
                class_name = class_names[idx] if idx < len(class_names) else f"Class {idx}"
                results_dict["class_metrics"][class_name] = float(ap)

        json_path = save_path / "evaluation_results.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False)

        print(f"\n✅ 결과 저장: {json_path}")

    return results


def compare_models(model_paths: list, data_path: str, split: str = "val"):
    """
    여러 모델의 성능을 비교합니다.

    Args:
        model_paths: 비교할 모델 경로 리스트
        data_path: 데이터셋 YAML 경로
        split: 평가 분할

    출력:
        모델별 성능 비교 표
        시각화 차트
    """
    print("=" * 70)
    print("모델 비교")
    print("=" * 70)

    results_list = []

    for model_path in model_paths:
        print(f"\n평가 중: {model_path}")
        model = YOLO(model_path)
        results = model.val(data=data_path, split=split, verbose=False)

        results_list.append({
            "name": Path(model_path).stem,
            "mAP@0.5": results.box.map50,
            "mAP@0.5:0.95": results.box.map,
            "precision": results.box.mp,
            "recall": results.box.mr
        })

    # 비교 표 출력
    print("\n" + "=" * 70)
    print("📊 모델 비교 결과")
    print("=" * 70)
    print(f"{'모델':<30s} {'mAP@0.5':<12s} {'mAP@0.5:0.95':<15s} {'Precision':<12s} {'Recall':<12s}")
    print("-" * 70)

    for r in results_list:
        print(f"{r['name']:<30s} {r['mAP@0.5']:<12.4f} {r['mAP@0.5:0.95']:<15.4f} {r['precision']:<12.4f} {r['recall']:<12.4f}")

    print("=" * 70)

    # 시각화
    try:
        import numpy as np

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Model Performance Comparison', fontsize=16)

        metrics = ['mAP@0.5', 'mAP@0.5:0.95', 'precision', 'recall']
        titles = ['mAP@0.5', 'mAP@0.5:0.95', 'Precision', 'Recall']

        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[idx // 2, idx % 2]
            names = [r['name'] for r in results_list]
            values = [r[metric] for r in results_list]

            ax.bar(names, values)
            ax.set_ylabel(title)
            ax.set_ylim([0, 1])
            ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300)
        print(f"\n✅ 비교 차트 저장: model_comparison.png")

    except Exception as e:
        print(f"\n⚠️ 시각화 생성 실패: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="YOLO 모델 평가 및 분석"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="평가할 모델 경로 (.pt 파일)"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="데이터셋 YAML 파일 경로"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["val", "test"],
        help="평가할 데이터 분할 (기본값: val)"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        help="결과 저장 디렉토리"
    )
    parser.add_argument(
        "--compare",
        nargs='+',
        help="비교할 추가 모델 경로들"
    )

    args = parser.parse_args()

    if args.compare:
        # 여러 모델 비교
        all_models = [args.model] + args.compare
        compare_models(all_models, args.data, args.split)
    else:
        # 단일 모델 평가
        evaluate_model(args.model, args.data, args.split, args.save_dir)


if __name__ == "__main__":
    main()
