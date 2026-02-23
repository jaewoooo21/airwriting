# 🎓 졸업작품 시연 시나리오 (Demo Script)

## 사전 준비 체크리스트
- [ ] ESP32 + 3 IMU 센서 하드웨어 장착 확인
- [ ] Wi-Fi 네트워크 연결 (ESP32 ↔ PC 같은 네트워크)
- [ ] Python 가상환경 활성화 및 의존성 설치 (`pip install -r requirements.txt`)
- [ ] 캘리브레이션 완료 (`python main.py --calibrate-only`)

---

## Step 1: 시스템 소개 (2분)
**보여줄 것**: 하드웨어 구성
- ESP32 (240MHz 듀얼코어) + WiFi UDP 네트워크
- S1 (전완, MPU-6050) + S2 (손등, MPU-6050) + S3 (펜, ICM-20948 9축)
- 펜 버튼 (GPIO 15) → 필기 시작/종료

## Step 2: 시스템 시작 (1분)
```
더블클릭: start_all.bat
```
3개 서비스가 자동 실행되고, 3초 후 웹 브라우저가 자동으로 열립니다.
- IMU Engine (UDP :12345)
- WebSocket Relay (WS :18765)
- Flask API (HTTP :5000)

## Step 3: 실시간 3D 궤적 시연 (3분)
**보여줄 탭**: 🏠 Desktop Studio
1. 연결 상태 확인 (우측 패널: 🟢 CONNECTED)
2. 펜을 잡고 허공에 글자를 써봄
3. **핵심 포인트**:
   - 3D 궤적이 실시간으로 캔버스에 그려짐
   - FK 좌표 (X/Y/Z) 실시간 업데이트
   - ZUPT 상태 표시 (정지 시 🟢)
   - S3 오일러각 실시간 파형
4. 카메라 조작: 드래그(회전), 스크롤(줌), V키(1인칭/3인칭 전환)

## Step 4: ML 학습 데모 (3분)
**보여줄 탭**: 🏠 Desktop Studio → 🧠 ML Training Lab
1. 좌측 패널 "🎯 [가이드] 단어 학습" 클릭
2. 모달에 글자 입력 (예: "A") → "3초 자동수집 시작"
3. 펜으로 A를 여러 번 써서 데이터 수집
4. 🧠 ML Training Lab 탭으로 이동 → 학습 통계 확인

## Step 5: 아키텍처 설명 (3분)
**보여줄 탭**: 📐 Architecture → 🔧 Hardware → 📊 Research
1. **Architecture 탭**: 시스템 데이터 파이프라인 다이어그램, ESKF 15-state, FK 체인, ZUPT 설명
2. **Hardware 탭**: ESP32 스펙, 센서 스펙, 통신 프로토콜
3. **Research 탭**: 인식률 비교 차트, 드리프트 그래프, Ablation Study, 참고문헌

## Step 6: 로드맵 (1분)
**보여줄 탭**: 🗺️ Roadmap
- Phase 1~3 완료 상태
- Phase 4~5 향후 계획

---

## ⚠️ 비상 대처 매뉴얼

### WS 연결이 안 될 때
```
1. web_relay.py가 실행 중인지 확인
2. 방화벽에서 포트 18765 허용
3. 브라우저 콘솔(F12) 에서 에러 확인
```

### ESP32 데이터가 안 올 때
```
1. ESP32의 Wi-Fi 연결 확인 (시리얼 모니터)
2. PC의 IP 주소 확인 (ipconfig)
3. python tools/health_check.py 실행
```

### 드리프트가 심할 때
```
1. python main.py --calibrate-only 재캘리브레이션
2. 센서가 단단히 고정되었는지 확인
3. 캘리브 중에는 센서를 완전히 정지
```

### Mock 모드 (하드웨어 없이 시연)
```
터미널 1: python tools/mock_esp32_imu.py
터미널 2: python main.py
터미널 3: python tools/web_relay.py
터미널 4: python web_app/app.py
→ http://localhost:5000
```
