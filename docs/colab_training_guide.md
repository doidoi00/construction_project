# Google Colab에서 YOLO11 DocLayNet 학습하기

## 🚀 빠른 시작

### 1단계: Colab 노트북 열기

1. [Google Colab](https://colab.research.google.com/) 접속
2. `파일` → `노트 업로드` → `notebooks/colab_train_doclaynet.ipynb` 업로드

### 2단계: GPU 설정

1. 메뉴: `런타임` → `런타임 유형 변경`
2. **하드웨어 가속기**: `T4 GPU` 선택 (무료)
3. `저장` 클릭

> **Colab Pro 사용자**: V100 또는 A100 선택 시 2배 빠름

### 3단계: 데이터셋 준비

#### 옵션 A: Google Drive에 업로드 (권장)

로컬에서 DocLayNet 데이터셋을 미리 다운로드 & 변환한 후 Drive에 업로드:

```bash
# 로컬에서 실행
# 1. DocLayNet 다운로드 및 변환
python src/data_preparation/download_doclaynet.py
python src/data_preparation/convert_to_yolo.py \
  --input data/doclaynet/raw \
  --output data/doclaynet/yolo

# 2. Google Drive에 업로드
# data/doclaynet/yolo 폴더를 Drive의 datasets/doclaynet로 업로드
```

#### 옵션 B: Colab에서 직접 다운로드

노트북의 "옵션 B: 직접 다운로드" 셀 실행 (2시간 소요)

### 4단계: 프로젝트 파일 업로드

#### 방법 1: ZIP 압축하여 업로드

```bash
# 로컬에서 실행
cd /Users/minyeop-jang/Documents/proj/yolo
zip -r yolo_project.zip . \
  -x "*.git*" "*__pycache__*" "*.pyc" \
  -x "data/doclaynet/*" "models/finetuned/*"
```

노트북에서 `files.upload()`로 업로드

#### 방법 2: GitHub 연동

```bash
# GitHub에 푸시 후 Colab에서 클론
!git clone https://github.com/your-username/yolo-project.git /content/yolo
```

### 5단계: 학습 실행

노트북의 "7. 학습 실행" 섹션에서 셀 실행:

```python
from ultralytics import YOLO

model = YOLO('yolo11n.pt')
results = model.train(
    data='data/doclaynet/yolo/doclaynet.yaml',
    epochs=100,
    batch=16,  # T4 GPU 기준
    imgsz=640,
    device=0,
    amp=True,  # CUDA FP16 지원!
    cache='ram',
)
```

---

## ⚡ Colab vs 로컬(MPS) 성능 비교

| 항목 | 로컬 (MPS) | Colab (T4 GPU) | Colab Pro (V100) |
|------|------------|----------------|------------------|
| **속도** | ~50ms/img | ~20ms/img | ~10ms/img |
| **배치 크기** | 2-4 | 16 | 32 |
| **FP16 지원** | ❌ | ✅ | ✅ |
| **100 에폭 소요시간** | ~20-25시간 | ~8-10시간 | ~4-6시간 |
| **비용** | 무료 | 무료 | $10/월 |
| **세션 제한** | 없음 | 12시간 | 24시간 |

> **결론**: Colab T4 GPU가 로컬 MPS보다 **2.5배 빠름**, Colab Pro V100은 **5배 빠름**

---

## 📊 최적화된 설정 (Colab 기준)

### T4 GPU (무료)

```python
epochs = 100
batch = 16        # 메모리: ~12GB
imgsz = 640
workers = 8
amp = True
cache = 'ram'
```

**예상 성능**:
- 에폭당 시간: ~5-6분
- 총 소요시간: ~8-10시간
- mAP@0.5: 85-90% (예상)

### V100 GPU (Colab Pro)

```python
epochs = 100
batch = 32        # 메모리: ~14GB
imgsz = 640
workers = 8
amp = True
cache = 'ram'
```

**예상 성능**:
- 에폭당 시간: ~2-3분
- 총 소요시간: ~4-6시간

### A100 GPU (Colab Pro+)

```python
epochs = 100
batch = 64        # 메모리: ~20GB
imgsz = 640
workers = 8
amp = True
cache = 'ram'
```

**예상 성능**:
- 에폭당 시간: ~1-2분
- 총 소요시간: ~2-3시간

---

## 🛡️ 세션 중단 대비

Colab 무료는 12시간 제한이 있으므로 체크포인트를 Drive에 저장:

```python
# 학습 설정에 추가
project = '/content/drive/MyDrive/yolo_project/models/finetuned'
save_period = 5  # 5 에폭마다 저장
```

### 학습 재개 방법

```python
# 중단된 지점부터 재개
from ultralytics import YOLO

resume_path = '/content/drive/MyDrive/yolo_project/models/finetuned/doclaynet_yolo11n_colab/weights/last.pt'
model = YOLO(resume_path)
results = model.train(resume=True)
```

---

## 🔧 문제 해결

### 1. GPU 메모리 부족 에러

**에러 메시지**:
```
RuntimeError: CUDA out of memory
```

**해결 방법**:
```python
# batch 크기 줄이기
batch = 8  # 16 → 8
# 또는 이미지 크기 줄이기
imgsz = 512  # 640 → 512
```

### 2. 데이터셋 경로 에러

**에러 메시지**:
```
FileNotFoundError: data/doclaynet/yolo/doclaynet.yaml not found
```

**해결 방법**:
```python
# 절대 경로 확인
!ls -la /content/yolo/data/doclaynet/yolo/

# doclaynet.yaml 파일 수정
# path: /content/yolo/data/doclaynet/yolo
```

### 3. Drive 마운트 실패

**해결 방법**:
```python
from google.colab import drive
drive.flush_and_unmount()  # 재마운트
drive.mount('/content/drive', force_remount=True)
```

### 4. 세션 타임아웃

**방법 1**: 브라우저 콘솔에서 스크립트 실행 (세션 유지)

```javascript
// 브라우저 개발자 도구 콘솔(F12)에서 실행
function ClickConnect(){
  console.log("클릭");
  document.querySelector("colab-connect-button").click()
}
setInterval(ClickConnect, 60000)
```

**방법 2**: Colab Pro 구독 (24시간 세션)

---

## 📈 학습 모니터링

### 실시간 로그 확인

노트북 셀 출력에서 확인:
```
Epoch    GPU_mem    box_loss    cls_loss    dfl_loss  Instances       Size
  1/100     4.23G      1.234       0.567       1.234        128        640
```

### TensorBoard 사용

```python
%load_ext tensorboard
%tensorboard --logdir /content/drive/MyDrive/yolo_project/models/finetuned
```

### 학습 그래프 확인

```python
from IPython.display import Image, display
display(Image('/content/drive/MyDrive/yolo_project/models/finetuned/doclaynet_yolo11n_colab/results.png'))
```

---

## 💾 모델 다운로드

### 방법 1: Google Drive에서 직접 다운로드

학습 완료 후 Drive에서 파일 다운로드:
- `models/finetuned/doclaynet_yolo11n_colab/weights/best.pt`

### 방법 2: Colab에서 직접 다운로드

```python
from google.colab import files
files.download('/content/drive/MyDrive/yolo_project/models/finetuned/doclaynet_yolo11n_colab/weights/best.pt')
```

### 방법 3: ZIP 압축하여 다운로드

```python
import shutil
from google.colab import files

shutil.make_archive('/content/trained_model', 'zip',
    '/content/drive/MyDrive/yolo_project/models/finetuned/doclaynet_yolo11n_colab')
files.download('/content/trained_model.zip')
```

---

## 🎯 로컬로 가져온 후 사용

```bash
# 로컬 프로젝트에 복사
cp best.pt /Users/minyeop-jang/Documents/proj/yolo/models/final/

# 추론 실행
python src/detection.py \
  --model models/final/best.pt \
  --source data/raw/sample_spec.pdf \
  --output data/output/
```

---

## 📚 추가 자료

- [YOLO 공식 문서](https://docs.ultralytics.com/)
- [DocLayNet 논문](https://arxiv.org/abs/2206.01062)
- [Colab Pro 가격](https://colab.research.google.com/signup)
- [문제 해결 가이드](https://github.com/ultralytics/ultralytics/issues)

---

## ✅ 체크리스트

학습 시작 전 확인:

- [ ] Colab GPU 설정 완료 (T4/V100/A100)
- [ ] Google Drive 마운트 완료
- [ ] DocLayNet 데이터셋 준비 완료
- [ ] 프로젝트 파일 업로드 완료
- [ ] doclaynet.yaml 경로 확인
- [ ] 학습 설정 (batch, imgsz, epochs) 확인
- [ ] Drive에 충분한 저장 공간 (최소 5GB)

학습 중 모니터링:

- [ ] GPU 메모리 사용률 (~80% 권장)
- [ ] 학습 로그 정상 출력
- [ ] loss 값 감소 확인
- [ ] 체크포인트 저장 확인 (5 에폭마다)

학습 완료 후:

- [ ] best.pt 모델 다운로드
- [ ] 학습 그래프 확인 (results.png)
- [ ] 검증 성능 확인 (mAP@0.5)
- [ ] 로컬에서 추론 테스트
