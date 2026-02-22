# AirWriting Unity 연동 설정 가이드

이 폴더에는 Python `airwriting_imu_only` 센서 퓨전 엔진에서 전송하는 UDP JSON 패킷을 수신하고, Unity 내부에서 3D로 시각화하는 데 필요한 C# 스크립트가 포함되어 있습니다.

## 1. 프로젝트 설정
1. 기존 Unity 3D 프로젝트를 열거나 새 프로젝트를 생성합니다.
2. `unity_integration/Scripts` 폴더를 Unity의 `Assets/` 프로젝트 패널로 드래그합니다.
3. Unity가 C# 스크립트를 컴파일할 때까지 기다립니다.

## 2. 기본 씬(Scene) 설정
1. 씬에 빈 GameObject를 생성하고 이름을 `AirWritingManager`로 지정합니다.
2. `AirWritingReceiver.cs` 스크립트를 이 `AirWritingManager`에 연결(Attach)합니다.
    - `Listen Port`를 확인하세요. 기본값은 `12346`이며, 이는 `system.yaml`의 `python_to_unity` 포트와 일치합니다.
3. `ArmController.cs` 스크립트를 동일한 `AirWritingManager`에 연결합니다. (필요한 컴포넌트로 자동 인식되어 수신기와 연결됩니다.)

## 3. 3D 팔 모델 생성
간단한 실린더(Cylinder) 모델을 사용하거나 다운로드한 리깅(Rigged) 팔 모델을 사용할 수 있습니다. 간단한 오브젝트를 사용하는 경우:
1. 3개의 GameObject(Capsule 또는 Cylinder)를 생성하고 다음과 같이 이름을 지정합니다:
   - `UpperArm` (위팔)
   - `Forearm` (아래팔)
   - `Hand` (손/펜)
2. 시각적으로 적절하게 배치합니다. 최상의 결과를 얻으려면 **피벗(Pivot)이 관절 위치(어깨, 팔꿈치, 손목)와 일치해야 합니다**. 계층 구조(UpperArm -> Forearm -> Hand)로 묶거나, 절대 회전값을 사용하므로 계층 없이 독립적으로 두어도 됩니다.
3. `AirWritingManager`에 연결된 `ArmController` 스크립트에서, 빈 슬롯(`UpperArm`, `Forearm`, `Hand`)에 방금 만든 3개의 오브젝트를 각각 드래그하여 할당합니다.

## 4. 펜 및 궤적(Trail) 설정
1. `Hand` GameObject 아래(자식)에 펜/손가락 끝을 나타내는 작은 구(Sphere)를 만들고 이름을 `PenTip`으로 지정합니다.
2. `PenTip`에 `TrailRenderer` 컴포넌트를 추가합니다.
   - 궤적이 남는 시간(Time)과 너비(Width)를 원하는 대로 설정합니다.
   - 궤적 색상(Color)을 설정합니다.
3. `PenTip` 오브젝트를 `ArmController`의 `PenTip`과 `TrailRenderer` 슬롯에 각각 드래그합니다.
4. (선택 사항) 2개의 Material(`WritingMat`, `IdleMat`)을 만들어 `ArmController`에 연결하면, ESP32 버튼(GPIO 15)을 누를 때 펜 색상이 변하는 시각적 피드백을 볼 수 있습니다.

## 5. 시스템 실행
1. Python 시스템을 실행합니다:
   ```bash
   python manage.py start
   ```
2. Python 터미널에 `Listening :12345 →Unity :12346 (IMU-only)` 및 `Calibration done` 메시지가 뜰 때까지 기다립니다.
3. Unity에서 **Play(재생)** 버튼을 누릅니다.
4. 센서를 움직여 봅니다. 펜 버튼(GPIO 15)을 누르고 있으면 Trail Renderer가 3D 공간에 그림을 그립니다!

## 6. 좌표계 문제 해결 (Troubleshooting)
만약 팔이 엉뚱한 축으로 움직인다면:
`AirWritingData.cs` 파일을 열어 `ParseQuaternion` 및 `ParseVector3` 메서드를 찾으세요.
IMU 센서가 팔에 부착된 물리적 방향의 차이와, Python의 오른손 좌표계(Right-handed)를 Unity의 왼손 좌표계(Left-handed)로 변환하는 과정에서 축을 바꾸거나 부호를 반전시켜야 할 때가 흔히 있습니다.
가상 팔 움직임이 실제 팔 움직임과 일치할 때까지 반환값(예: `new Quaternion(-y, x, -z, w)`)의 매핑을 수정해 보세요.
