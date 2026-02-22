---
name: python-airwriting-review
description: AirWriting 프로젝트의 Python 코드(센서 퓨전, IMU 데이터 처리, 좌표계)를 리뷰하고 가이드하는 스킬입니다.
version: 2.0.0
author: antigravity
tags:
  - python
  - sensor-fusion
  - imu
  - airwriting
  - research
  - automation
language: ko
---

# AirWriting Python Code Review

## Purpose

이 스킬은 AirWriting 프로젝트의 핵심인 **IMU 센서 퓨전, 좌표계 변환, 데이터 처리 로직**의 정확성과 일관성을 유지하기 위해 사용합니다. **자동화된 검증(Orchestrator)**과 **최신 연구(RESEARCH.md)**에 기반한 설계를 강제합니다.

## When to use this skill

- Python 코드를 수정하거나 새로 작성할 때 (특히 `airwriting_imu/fusion`, `core`, `filters` 패키지).
- 센서 데이터 처리, 좌표계 변환, 칼만 필터 로직을 건드릴 때.
- **새로운 알고리즘이나 필터를 도입할 때 (반드시 `RESEARCH.md`와 대조).**
- 사용자가 "코드 리뷰해줘", "이 로직 맞는지 봐줘"라고 요청했을 때.

## How to use it

1.  **자동화 검증 (Mandatory)**: 코드를 수정하기 전후에 반드시 `python tools/orchestrator.py`를 실행하여 기존 테스트와 물리학적 일관성 체크를 통과하는지 확인합니다.
2.  **연구 기반 설계**: 새로운 기능을 추가할 때는 `RESEARCH.md`를 참조하여 SOTA(State-of-the-Art) 방식과 일치하는지, 혹은 그에 준하는 근거가 있는지 확인합니다.
3.  **Checklist 대조**: 아래 체크리스트를 기반으로 코드를 검토합니다.
4.  **구체적 피드백**: 문제가 있는 부분은 정확한 라인 번호와 함께 수정 제안을 합니다.

## Checklist

### 1. Automation & Verification (NEW)
- [ ] **Orchestrator Pass**: `python tools/orchestrator.py`가 모든 테스트를 통과했습니까?
- [ ] **Type Checking**: 주요 함수에 Type Hinting이 완벽하게 되어 있고, `mypy` 오류가 없습니까?
- [ ] **Physics Consistency**: 에너지 보존 법칙이나 kinematic constraints(예: 팔 길이 제한)가 시뮬레이션 상에서 위배되지 않는지 확인했습니까?

### 2. Research & Algorithm Alignment (NEW)
- [ ] **Deep Learning Integration**: ZUPT나 Drift Correction에 ML 모델(`airwriting_imu/ml` 참조)을 활용하는 방안을 고려했습니까? (`RESEARCH.md` 참조)
- [ ] **SOTA Comparison**: 구현하려는 로직이 최신 연구(2024-2025) 트렌드와 역행하지 않는지 확인했습니까?

### 3. Coordinate Systems & Math
- [ ] **쿼터니언 순서**: `[w, x, y, z]` 순서를 따르는지 확인합니다. (NumPy/SciPy 관례 확인)
- [ ] **회전 행렬 곱**: `R_world_to_body` vs `R_body_to_world`가 혼동되지 않았는지 확인합니다.
- [ ] **Bone Direction**: Forward Kinematics에서 기본 Bone 방향이 `[1, 0, 0]` (X축)으로 가정되어 있는지 확인합니다.
- [ ] **단위**: 거리(m), 속도(m/s), 가속도(m/s²), 각속도(rad/s), 시간(초/마이크로초) 단위가 일관적인지 확인합니다.

### 4. Sensor Fusion Logic (ESKF)
- [ ] **State Vector**: 15-state 정의(Pos, Vel, Att, Bg, Ba)가 유지되는지 확인합니다.
- [ ] **Bias 처리**: 가속도/자이로 바이어스가 적절히 피드백(Nominal State 보정)되는지 확인합니다.
- [ ] **ZUPT/ZARU**: 정지 상태 감지 로직(SHOE 에너지 디텍터 및 Neural Network)이 적절한지 확인합니다.
- [ ] **Accel Bias**: `_predict_nominal()`에서 `accel_w - self.ba`로 바이어스가 빠지는지 확인합니다. (v2.0 필수)
- [ ] **F Matrix**: `_predict_error_cov()`에서 `omega = gyro - self.bg`를 사용하는지 확인합니다. (v2.0 필수)
- [ ] **Adaptive Q**: 정지 시 Q 축소, 운동 시 Q 확대가 적용되는지 확인합니다.

## Feedback Loop
모든 변경 사항은 **자동화된 테스트 + 사용자 리뷰 + 연구 문서 업데이트**의 3단계 피드백 루프를 거쳐야 합니다.

## Examples

### 좋은 피드백 예시
> "새로 추가하신 ZUPT 로직은 `RESEARCH.md`에 언급된 'Neural ZUPT' 방식과 유사하지만, 임계값이 너무 낮아 오작동할 수 있습니다. `orchestrator.py` 실행 결과 테스트 3번이 실패했습니다. 임계값을 `0.2`로 조정하고 다시 테스트해 주세요."

### 나쁜 피드백 예시
> "ZUPT 코드가 이상합니다. 고쳐주세요." (구체적인 이유와 근거 부족)

## Notes
- 이 프로젝트는 **IMU-Only** 구성을 지향하므로 UWB 관련 로직이 섞여 들어가지 않도록 주의합니다.
- 파일 수정 시 불필요한 공백 변경을 최소화하여 Diff를 깔끔하게 유지합니다.
