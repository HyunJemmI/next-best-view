# next-best-view

macOS에서 `MuJoCo + Open3D + PyTorch` 기반의 active perception MVP를 빠르게 검증하고, 이후 `3D Gaussian Splatting semantic representation + language-conditioned NBV + arm-aware planning`으로 확장하기 위한 연구용 저장소입니다.

현재 저장소는 최종 연구 스택 전체를 다 구현한 상태는 아니고, 먼저 아래 루프가 실제로 동작하는 최소 실험 환경을 제공합니다.

`render -> RGB-D -> point cloud -> Gaussian proxy semantic state -> uncertainty / language affinity -> NBV selection -> repeat`

## 현재 상태

현재 확인된 항목:
- macOS에서 `mujoco`, `open3d`, `torch` import 성공
- MuJoCo offscreen rendering 성공
- RGB-D를 world-frame point cloud로 복원 성공
- Gaussian proxy global map 누적 성공
- PyTorch 기반 mock semantic inference 성공
- multi-iteration NBV loop 실행 성공
- iteration별 PNG / PLY / JSON 로그 저장 성공

현재는 아직 mock 단계인 항목:
- 실제 LightSplat / LangSplat / SAM / CLIP 모델 연결
- category prior bank
- grasp-aware `AffIG`
- arm reachability / IK / collision

## 저장소 목표

이 저장소는 아래 두 단계를 잇는 중간 기반으로 사용합니다.

1. `MVP`
- free-camera 기반 active perception loop 검증
- Gaussian semantic state와 NBV 점수화 인터페이스 고정
- macOS 환경에서 라이브러리 동작 여부 확인

2. `확장 단계`
- 실제 3DGS semantic checkpoint 연결
- CLIP 기반 language query / target instance 연결
- category prior / shape prior 활용
- arm-mounted camera view planning
- grasp-aware view utility와 reachability 통합

## 디렉토리 구조

```text
next-best-view/
├─ README.md
├─ requirements.txt
├─ setup.sh
├─ run_demo.py
├─ configs/
│  └─ default.yaml
├─ docs/
│  ├─ mvp_plan.md
│  └─ notion_execution_plan.md
├─ assets/
│  ├─ mujoco/
│  │  └─ scene.xml
│  └─ meshes/
├─ checkpoints/
├─ outputs/
│  ├─ logs/
│  ├─ pcd/
│  └─ debug/
└─ src/
   ├─ sim/
   ├─ models/
   ├─ perception/
   ├─ planning/
   └─ utils/
```

## 빠른 시작

### 1. 환경 설치

Python 3.11 기준으로 맞춰 두었습니다.

```bash
bash setup.sh
```

### 2. 환경 sanity check

```bash
.venv/bin/python run_demo.py --check-env
```

이 명령은 아래를 빠르게 확인합니다.
- 패키지 import
- Torch backend 사용 가능 여부
- MuJoCo scene load
- 1회 렌더링
- point cloud 복원 가능 여부

### 3. 데모 실행

```bash
.venv/bin/python run_demo.py
```

## 주요 산출물

- `outputs/debug/`
  - iteration별 RGB
  - depth preview
  - top-down map snapshot
  - candidate score plot
- `outputs/pcd/`
  - iteration별 fused global map `.ply`
- `outputs/logs/run_summary.json`
  - 시점 선택과 상태 요약 로그

## 구현 개요

### 시뮬레이션
- `src/sim/mujoco_env.py`
  - MuJoCo scene 로드, 카메라 pose 설정, RGB-D 렌더링
- `src/sim/camera_utils.py`
  - orbit / look-at pose 생성

### Perception
- `src/perception/pointcloud.py`
  - RGB-D -> world point cloud
- `src/perception/gaussian_state.py`
  - voxelized Gaussian proxy state
- `src/perception/fusion.py`
  - global map 누적
- `src/perception/language_features.py`
  - 현재는 toy language embedding encoder

### Models
- `src/models/gaussian_semantic_wrapper.py`
  - 현재는 mock semantic inference
  - 향후 실제 LightSplat 계열 래퍼로 교체 예정

### Planning
- `src/planning/view_sampler.py`
  - candidate orbit view 생성
- `src/planning/score_terms.py`
  - `DeltaU`, `LangAffinity`, `ConsistencyGain`, `OcclRelief`, `MoveCost`
- `src/planning/nbv.py`
  - 후보 점수 계산과 다음 시점 선택
- `src/planning/reachability.py`
  - 현재는 arm feasibility stub

## 문서

- [MVP Plan](docs/mvp_plan.md)
- [Notion Execution Plan](docs/notion_execution_plan.md)

## 다음 작업 우선순위

1. `GaussianSemanticWrapper`를 실제 LightSplat/LangSplat 계열 모델로 교체
2. CLIP text/image embedding을 실제 feature pipeline에 연결
3. category prior / shape prior 기반 score term 추가
4. `ReachabilityEvaluator`에 로봇팔 IK / collision 검사를 연결
5. free-camera 후보를 arm-mounted camera 후보로 치환
