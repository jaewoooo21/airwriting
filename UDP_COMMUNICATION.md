# 📡 AirWriting UDP 통신 구조 레퍼런스

> **⚠️ 중요: 이 문서는 현재 작동하는 통신 구조의 정확한 기록입니다.**
> **포트, 패킷 포맷, JSON 필드를 변경하면 시스템이 깨질 수 있습니다.**

---

## 전체 통신 아키텍처

```
┌──────────┐   UDP :12345    ┌───────────────┐   UDP :12346   ┌──────────────┐   WS :18765   ┌─────────┐
│  ESP32   │ ──────────────→ │   main.py     │ ─────────────→ │ web_relay.py │ ────────────→ │ Browser │
│ (Wi-Fi)  │  Binary Packet  │ (controller)  │  JSON String   │ (WebSocket)  │  JSON String  │ (app.js)│
└──────────┘                 └───────────────┘                └──────────────┘               └─────────┘
                                    │
                                    │ UDP :12347 (JSON)
                                    ↓
                              ┌──────────────┐
                              │  (Dashboard)  │
                              │  (미사용/예비) │
                              └──────────────┘
```

---

## 1. 포트 정리 (절대 변경 금지)

| 포트 | 프로토콜 | 방향 | 설정 위치 | 용도 |
|------|----------|------|-----------|------|
| **12345** | UDP | ESP32 → Python | `system.yaml` → `network.ports.esp32_to_python` | IMU 바이너리 패킷 수신 |
| **12346** | UDP | Python → web_relay | `system.yaml` → `network.ports.python_to_unity` | 처리된 JSON 프레임 송신 |
| **12347** | UDP | Python → Dashboard | `system.yaml` → `network.ports.python_to_dashboard` | 원시 센서 데이터 (예비) |
| **18765** | WebSocket | web_relay → Browser | `web_relay.py` → `WS_PORT` / `app.js` → `WS_URL` | 웹 대시보드 실시간 스트림 |
| **5000** | HTTP | Browser → Flask | `app.py` | 웹 대시보드 & ML API |

### 설정 파일 위치
```yaml
# config/system.yaml (lines 3-14)
network:
  ports:
    esp32_to_python: 12345    # ← ESP32 펌웨어에도 동일하게 설정 필수
    python_to_unity: 12346    # ← web_relay.py UDP_PORT와 일치 필수
    python_to_dashboard: 12347
  unity:
    ip: "127.0.0.1"
    rate_hz: 60
  dashboard:
    ip: "127.0.0.1"
    rate_hz: 10
```

```python
# tools/web_relay.py (lines 17-20) — 반드시 system.yaml과 일치
UDP_PORT = 12346    # ← system.yaml의 python_to_unity와 동일
WS_PORT = 18765     # ← app.js의 WS_URL과 동일
```

```javascript
// web_app/app.js (line 11)
const WS_URL = `ws://${window.location.hostname}:18765`;  // ← web_relay.py WS_PORT와 동일
```

---

## 2. ESP32 → Python 패킷 포맷 (바이너리)

`controller.py`의 `_parse()` 메서드에서 처리. 3가지 버전 지원:

### V3 (현재 사용, 92 바이트) — 9축 S3 포함
```
[0xAA] [4B timestamp] [S1: 24B] [S2: 24B] [S3: 36B] [1B button] [1B checksum] [0x55]
 ↑                      ↑          ↑          ↑         ↑           ↑            ↑
 Header   uint32 μs    6×float   6×float    9×float   GPIO15      XOR(1..89)   Footer
                       ax,ay,az   ax,ay,az   ax,ay,az
                       gx,gy,gz   gx,gy,gz   gx,gy,gz
                                             mx,my,mz
```

| 바이트 오프셋 | 크기 | 내용 |
|-------------|------|------|
| 0 | 1B | Header `0xAA` |
| 1-4 | 4B | Timestamp (uint32, microseconds) |
| 5-28 | 24B | S1 (전완, MPU-6050): `<6f` → [ax,ay,az,gx,gy,gz] |
| 29-52 | 24B | S2 (손등, MPU-6050): `<6f` → [ax,ay,az,gx,gy,gz] |
| 53-88 | 36B | S3 (펜, ICM-20948): `<9f` → [ax,ay,az,gx,gy,gz,mx,my,mz] |
| 89 | 1B | Button (bit 0 = pen down) |
| 90 | 1B | Checksum (XOR of bytes 1..89) |
| 91 | 1B | Footer `0x55` |

### 축 매핑 (controller.py에서 적용)
```python
# 모든 센서 동일: 펌웨어 x=-x, y=-y 반영
a = np.array([-raw[0], -raw[1], raw[2]])   # x=-x, y=-y, z=z
g = np.array([-raw[3], -raw[4], raw[5]])   # 동일
m = np.array([-raw[6], -raw[7], raw[8]])   # S3만 (9축)
```

### V2 (80 바이트) — 버튼 포함, S3도 6축
```
[0xAA] [4B ts] [S1: 24B] [S2: 24B] [S3: 24B] [1B btn] [1B cksum] [0x55]
                                                         XOR(1..77)
```

### V1 (79 바이트) — 버튼 없음
```
[0xAA] [4B ts] [S1: 24B] [S2: 24B] [S3: 24B] [1B cksum] [0x55]
                                                XOR(1..76)
```

---

## 3. Python → web_relay JSON 페이로드

`controller.py`의 `_flush_unity()` 메서드에서 생성, `_tx_sock`으로 **포트 12346**에 전송.

```json
{
  "t": "f",
  "ms": 1708700000000,
  "S1q": [1.0, 0.0, 0.0, 0.0],
  "S1e": [0.0, 0.0, 0.0],
  "S2q": [1.0, 0.0, 0.0, 0.0],
  "S2e": [0.0, 0.0, 0.0],
  "S3q": [0.998, 0.01, -0.02, 0.03],
  "S3e": [5.2, -10.1, 3.8],
  "S3p": [0.015, 0.42, 0.12],
  "S3v": [0.001, -0.002, 0.0],
  "S3z": false,
  "S3zaru": false,
  "pen": true,
  "S3fk": [0.012, 0.41, 0.11]
}
```

| 필드 | 타입 | 설명 |
|------|------|------|
| `t` | string | 프레임 타입 (`"f"` = frame) |
| `ms` | int | Unix timestamp (ms) |
| `S1q`, `S2q`, `S3q` | float[4] | 쿼터니언 [w,x,y,z] (Hamilton) |
| `S1e`, `S2e`, `S3e` | float[3] | 오일러각 [pitch,roll,yaw] (degrees) |
| `S3p` | float[3] or null | ESKF 위치 [x,y,z] (meters) |
| `S3v` | float[3] or null | ESKF 속도 [vx,vy,vz] (m/s) |
| `S3z` | bool | ZUPT 활성 여부 |
| `S3zaru` | bool | ZARU 활성 여부 |
| `pen` | bool | 펜 버튼 상태 (true=필기 중) |
| `S3fk` | float[3] or null | FK 펜끝 좌표 [x,y,z] |

> **`app.js`는 이 JSON의 `S1q`, `S2q`, `S3q`로 FK를 재계산하고, `pen`으로 필기 상태를 판단합니다.**

---

## 4. 소켓 바인딩 상세

### controller.py (main.py → start)
```python
# 수신 (ESP32로부터)
self._rx_sock = socket.socket(AF_INET, SOCK_DGRAM)
self._rx_sock.bind(("0.0.0.0", 12345))   # 모든 인터페이스에서 수신
self._rx_sock.settimeout(0.5)

# 송신 (Unity/web_relay로)
self._tx_sock = socket.socket(AF_INET, SOCK_DGRAM)
# bind 없음 — sendto()로 12346, 12347에 전송
```

### web_relay.py
```python
# UDP 수신 (main.py로부터)
sock = socket.socket(AF_INET, SOCK_DGRAM)
sock.bind(("0.0.0.0", 12346))   # python_to_unity 포트 수신

# WebSocket 서버
ws_server = websockets.serve(ws_handler, "0.0.0.0", 18765)
# 브라우저 → ws://localhost:18765 로 연결
```

---

## 5. 통신 트러블슈팅

### 증상별 해결법

| 증상 | 원인 | 해결 |
|------|------|------|
| 웹 대시보드 `CONNECTING...` | web_relay.py 미실행 | `python tools/web_relay.py` 실행 |
| `CONNECTING...` 지속 | 포트 18765 방화벽 차단 | Windows 방화벽에서 허용 |
| web_relay 실행되지만 데이터 없음 | main.py 미실행 | `python main.py` 먼저 실행 |
| main.py `Listening :12345` 후 데이터 없음 | ESP32 미연결/IP 불일치 | ESP32 시리얼 모니터에서 연결 IP 확인 |
| `WinError 10048` (포트 사용 중) | 이전 프로세스가 포트 점유 | `netstat -ano | findstr :12345` 로 PID 확인 후 종료 |
| `WinError 10054` (연결 거부) | 수신측 미실행 상태에서 송신 | 정상 동작 (UDP 특성), 무시 가능 |

### 포트 점유 확인 명령어
```powershell
# 12345 포트 사용 중인 프로세스 확인
netstat -ano | findstr :12345

# 프로세스 강제 종료
taskkill /PID <PID> /F

# 한번에 모든 관련 포트 정리 (start_all.bat에 포함됨)
for /f "tokens=5" %a in ('netstat -aon ^| findstr :12345') do taskkill /PID %a /F
```

### 실행 순서 (중요)
```
1. python main.py              ← 12345 수신 시작, 12346/12347 송신 시작
2. python tools/web_relay.py   ← 12346 수신, 18765 WS 서버 시작
3. python web_app/app.py       ← 5000 HTTP 서버 시작
4. 브라우저 → http://localhost:5000 → ws://localhost:18765 연결
```
> `start_all.bat`은 이 순서를 자동으로 실행합니다.

---

## 6. Mock 테스트 (하드웨어 없이)

```powershell
# 터미널 1: 가짜 ESP32 데이터 생성 → 12345로 전송
python tools/mock_esp32_imu.py

# 터미널 2: 퓨전 엔진
python main.py

# 터미널 3: WebSocket 릴레이
python tools/web_relay.py

# 터미널 4: 웹 서버
python web_app/app.py
```
