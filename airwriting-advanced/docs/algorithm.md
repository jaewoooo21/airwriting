# Pipeline notes (what this code is doing)

## Frames / quaternions
- Quaternion order: `[w, x, y, z]`
- `q_WX` means: frame X -> frame W (vector in X rotated to W).

VQF returns orientation quaternions (6D/9D). This project treats that as `q_WS` (sensor -> world).
Then mount correction is applied:
- `q_WG = q_WS ⊗ q_SG`
- 여기서 `q_SG`는 “세그먼트(팔 뼈 프레임) -> 센서 프레임” 고정 회전(마운트)이다.
- 센서 프레임이 세그먼트 프레임이랑 거의 같으면 그냥 `[1,0,0,0]`로 두면 됨.

## Kinematic chain (3-link)
- Shoulder(원점)
- Elbow = Shoulder + R(q_WU)*axis*L1
- Wrist = Elbow + R(q_WF)*axis*L2
- Tip = Wrist + R(q_WH)*axis*L3

`axis`는 각 세그먼트 프레임에서 “뼈가 향하는 방향”으로 잡는다.
보통 strap 방향이 +X라면 `[1,0,0]`.

## Joint constraints
- Elbow를 **hinge**로 보고, 상대회전 `q_UF`를 swing/twist로 분해해서 **twist(힌지 축 회전)**만 남기는 방식.
- Wrist는 완전 hinge로 박아버리면 필기 동작이 죽어서,
  swing 각도만 제한(soft limit)하고 strength로 부드럽게 걸어놓음.

이게 “yaw 드리프트” 같은 누적 오차를 강제로 제약 안으로 눌러넣는 역할을 해줌.

## Writing plane projection
- 펜팁 3D `tip_w`를 평면으로 정사영하고,
- 평면 basis (u,v)로 2D 좌표 `uv`로 바꾼다.

## Stroke detection
- uv 속도로 pen-down/up을 대충 판단한다.
- speed hysteresis + 최소 다운시간(min_down_time) 들어가있음.
