# MVP 정의

## 목표

macOS에서 다음 경로가 실제로 돌아가는 최소 연구 프로토타입을 만든다.

`MuJoCo 렌더링 -> RGB-D -> world point cloud -> Gaussian proxy state -> language-aware uncertainty -> NBV -> 반복`

## 이 MVP에서 검증하는 것

- MuJoCo offscreen rendering이 macOS에서 안정적으로 되는가
- depth를 point cloud로 복원했을 때 geometry가 정상적인가
- PyTorch가 `mps` 또는 `cpu`에서 inference 역할을 수행하는가
- Gaussian proxy state가 view 누적에 따라 커지는가
- language-aware score를 사용해 plausible한 다음 시점을 고를 수 있는가

## 이 MVP에서 의도적으로 미루는 것

- 실제 LightSplat/LangSplat checkpoint
- 실제 SAM / CLIP 모델 추론
- 로봇팔 IK / collision
- grasp execution

## 설계 요점

- 현재 `Gaussian`은 실제 3DGS parameter가 아니라 voxelized proxy state이다.
- planner는 free-camera로 실행하지만 candidate pose는 나중에 end-effector pose로 연결할 수 있게 설계한다.
- mock semantic wrapper는 `target language affinity`, `uncertainty`, `reliability`를 반환하여 실제 semantic model의 입출력 계약을 먼저 고정한다.

## 이후 파이프라인

1. `GaussianSemanticWrapper`를 실제 LightSplat 기반 모델로 교체
2. query text를 실제 CLIP embedding으로 바꾸고 instance CLIP feature와 연결
3. `score_terms.py`에 `AffIG`, `PartAlign`, `OcclRelief`를 task-aware로 확장
4. `ReachabilityEvaluator`에 로봇팔 reachable pose / collision check를 연결
5. free-camera 후보를 arm-mounted camera 후보로 치환
