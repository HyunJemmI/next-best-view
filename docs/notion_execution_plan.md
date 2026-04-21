# Notion 정리용 실행 계획

## 0. 프로젝트 목표

- macOS에서 돌아가는 active perception 연구용 MVP 구축
- MuJoCo, Open3D, PyTorch가 실제로 한 루프 안에서 연결되는지 검증
- 이후 3DGS semantic model과 arm-aware NBV로 확장 가능한 구조 확보

## 1. 이번 스프린트 산출물

- [ ] Python 3.11 가상환경 구축
- [ ] MuJoCo 장면 로드 및 RGB-D 렌더링 성공
- [ ] RGB-D -> point cloud 복원 성공
- [ ] Gaussian proxy global map 누적 성공
- [ ] PyTorch 기반 mock semantic inference 성공
- [ ] language-aware NBV 후보 점수 계산 성공
- [ ] multi-iteration demo 실행 성공
- [ ] debug 이미지 / point cloud / 로그 저장 성공

## 2. 단계별 수행 내용

### 단계 A. 환경 구축

- [ ] `python3.11` 설치
- [ ] `.venv` 생성
- [ ] `requirements.txt` 설치
- [ ] `python run_demo.py --check-env` 통과

완료 기준:
- MuJoCo / Open3D / Torch import 성공
- Torch backend 확인 가능

### 단계 B. 렌더링 검증

- [ ] MuJoCo scene.xml 작성
- [ ] 카메라 rig pose 제어 구현
- [ ] RGB 저장
- [ ] depth preview 저장

완료 기준:
- 시점 변경 시 이미지가 실제로 바뀜
- depth map이 비어 있지 않음

### 단계 C. 기하 복원 검증

- [ ] depth backprojection 구현
- [ ] world-frame point cloud 생성
- [ ] Open3D `.ply` 저장

완료 기준:
- target/table/occluder가 point cloud에서 구분됨
- 장면 스케일이 비정상적이지 않음

### 단계 D. Semantic proxy state

- [ ] Gaussian proxy state 자료구조 구현
- [ ] view별 observation 누적
- [ ] uncertainty / reliability 필드 누적

완료 기준:
- iteration이 늘수록 proxy 수 또는 coverage가 증가함

### 단계 E. Mock semantic model

- [ ] text embedding encoder 구현
- [ ] PyTorch mock inference 구현
- [ ] logits / predictions / uncertainty / language similarity 반환

완료 기준:
- device가 `mps` 또는 `cpu`로 선택됨
- output shape이 point 수와 일치함

### 단계 F. NBV loop

- [ ] orbit candidate sampling
- [ ] score term 계산
- [ ] 다음 시점 선택
- [ ] iteration 로그 저장

완료 기준:
- random이 아닌 규칙 기반 시점 선택이 수행됨
- 후보별 점수 breakdown을 볼 수 있음

## 3. 다음 스프린트

- [ ] 실제 LightSplat semantic checkpoint 연결
- [ ] CLIP text/image embedding으로 교체
- [ ] category prior bank 설계
- [ ] OcclRelief와 ConsistencyGain 정교화
- [ ] 로봇팔 reachability layer 연결
