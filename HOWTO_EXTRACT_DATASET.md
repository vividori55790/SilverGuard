# 다른 컴퓨터(Docker)에서 YOLO11 Pose 데이터셋 추출 가이드

이 문서는 다른 컴퓨터의 **Docker** 환경에서 `yolo11s-pose` 모델을 사용하여 비디오로부터 **Track ID**를 포함한 포즈 데이터셋을 추출하는 방법을 설명합니다.

사용해야 할 코드는 **`preprocess_all.py`** 입니다.

---

## 1. 사전 준비 (Prerequisites)

대상 컴퓨터에 **Docker**가 설치되어 있어야 합니다. GPU를 사용하려면 **NVIDIA Container Toolkit**도 필요합니다.

### Docker 이미지 다운로드

Ultralytics 공식 이미지를 사용합니다.

```bash
docker pull ultralytics/ultralytics:latest
```

## 2. 프로젝트 폴더 및 데이터 준비

데이터 추출 코드는 `preprocess_all.py`를 사용합니다. 이 코드는 다음과 같은 경로 구조를 가정합니다.

- **코드 경로**: `/app/SilverGuard/preprocess_all.py`
- **모델 경로**: `/app/SilverGuard/models/yolo11s-pose.pt`
- **데이터(비디오) 경로**: `/app/data` (이 폴더 아래의 모든 동영상을 재귀적으로 탐색합니다)
- **결과 저장 경로**: `/app/data/raw_keypoints_all.csv`

## 3. Docker 컨테이너 실행 명령어

호스트 컴퓨터의 폴더를 컨테이너 내부에 마운트하여 환경을 구성합니다.

### 명령어 구조

```bash
docker run -it --ipc=host --gpus all \
  -v "호스트_코드_경로:/app/SilverGuard" \
  -v "호스트_비디오_경로:/app/data" \
  ultralytics/ultralytics:latest
```

- `--gpus all`: GPU 사용 시 필수 (CPU만 사용한다면 제외)
- **호스트*코드*경로**: `preprocess_all.py` 파일과 `models` 폴더가 들어있는 폴더 경로
- **호스트*비디오*경로**: 분석할 동영상 파일들이 들어있는 폴더

## 4. 데이터 추출 실행

컨테이너 내부로 진입한 후(위 명령어를 치면 자동으로 진입됨), 다음 명령어를 실행합니다.

```bash
# 1. 필요한 패키지 확인 (tqdm이 없다면 설치)
pip install tqdm

# 2. 스크립트 실행
python /app/SilverGuard/preprocess_all.py
```

### 주의사항 (코드 수정 필요 시)

`preprocess_all.py`의 93번째 줄에 `device='cpu'`로 설정되어 있을 수 있습니다. GPU를 사용하려면 이 부분을 삭제하거나 `device=0`으로 변경하세요.

```python
# 수정 전 (CPU 강제)
results = model.track(..., device='cpu')

# 수정 후 (GPU 사용 또는 자동 선택)
results = model.track(..., device=0)
# 또는
results = model.track(...)
```

## 5. 결과 확인

작업이 완료되면 호스트 컴퓨터의 **비디오 폴더** 내에 `raw_keypoints_all.csv` 파일이 생성됩니다.

### CSV 데이터 형식

| video_name | frame_idx | time_sec | track_id | x0   | y0   | c0   | ... | x16  | y16  | c16  |
| :--------- | :-------- | :------- | :------- | :--- | :--- | :--- | :-- | :--- | :--- | :--- |
| cam1.mp4   | 1         | 0.033    | 1        | 0.51 | 0.32 | 0.98 | ... | 0.55 | 0.88 | 0.85 |

- `track_id`: 동일 인물 추적 ID
- `x, y`: 이미지 내 정규화된 좌표 (0.0 ~ 1.0)
- `c`: 신뢰도 점수 (Confidence)

## 6. AI 에이전트 위임용 프롬프트 (Prompt for AI Agent)

다른 AI(ChatGPT, Claude 등)에게 이 작업을 위임하거나 가이드를 요청할 때 바로 사용할 수 있는 프롬프트입니다.

---

**[Prompt 내용 복사 시작]**

**목표**: `Docker` 환경에서 `yolov11s-pose` 모델을 사용하여 비디오 데이터로부터 **Track ID가 포함된 Keypoints Dataset(CSV)**을 추출해야 한다.

**환경 설정 정보**:

1.  **OS**: Windows 또는 Linux (Docker 설치됨)
2.  **Docker Image**: `ultralytics/ultralytics:latest`
3.  **필수 파일**:
    - 소스 코드: `SilverGuard/preprocess_all.py` (이 파일은 `/app/SilverGuard`에 마운트됨)
    - 데이터(영상): `/app/data` (호스트의 영상 폴더가 이 경로로 마운트됨)

**작업 순서 명령**:

1.  호스트의 `SilverGuard` 폴더와 `Video` 폴더 경로를 확인해라.
2.  다음 Docker 실행 명령어를 생성하고 사용자에게 실행하라고 안내해라 (GPU 사용 옵션 포함):
    - `docker run -it --ipc=host --gpus all -v [HOST_CODE_PATH]:/app/SilverGuard -v [HOST_VIDEO_PATH]:/app/data ultralytics/ultralytics:latest`
3.  컨테이너 내부 진입 후 다음 명령어를 순차적으로 실행하는 가이드를 제공해라:
    - `pip install tqdm` (진행률 표시 라이브러리 설치)
    - `python /app/SilverGuard/preprocess_all.py`
4.  만약 `device='cpu'` 관련 에러나 느린 속도 문제가 발생하면, 파이썬 스크립트 내 `model.track(...)` 부분에서 `device=0`으로 변경하도록 안내해라.
5.  최종 결과물인 `raw_keypoints_all.csv`가 호스트의 비디오 데이터 폴더에 생성됨을 확인시켜라.

이 과정을 수행하기 위한 구체적인 터미널 명령어와 절차를 알려줘.

## **[Prompt 내용 복사 끝]**
