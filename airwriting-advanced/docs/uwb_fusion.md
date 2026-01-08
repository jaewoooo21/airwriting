# UWB (2-module) extension hook

이 코드베이스는 기본은 IMU-only로 3D tip을 뽑는다.

UWB를 붙일 때 현실적인 역할은 보통 둘 중 하나임:
1) **tip(또는 손목) 위치에 대한 range constraint**: |p_tip - p_anchor| = r  
2) **heading/yaw 안정화**: IMU yaw 드리프트를 range + kinematic chain으로 완만하게 보정

현재 `src/airwriting/sensors/uwb_serial.py`는 range를 읽기만 하고,
fusion은 “어디에 넣을지”가 프로젝트마다 달라서 코드에는 훅만 남겨놨다.

추천 구현(가장 쉬운 버전):
- 고정 anchor 1개 + 손에 tag 1개(2모듈)
- 매 프레임:
  - IMU로 tip 위치 p_tip(shoulder frame) 계산
  - world에서 shoulder 위치/heading을 (x,y,yaw) 3자유도로만 둔다고 가정
  - UWB range residual e = ||Rz(yaw)*p_tip + t_xy - p_anchor|| - r
  - 작은 Gauss-Newton 3변수 최적화 1~2 iter 돌려서 (x,y,yaw) 업데이트
  - 결과로 world 좌표 tip이 UWB에 맞게 따라감

이걸 넣으면 “필기 평면이 world 기준으로 돌아가는 현상”이 확 줄어듦.
