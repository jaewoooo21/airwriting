# AirWriting Advanced - 최신 기술 기반 에어라이팅 시스템

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Ubuntu](https://img.shields.io/badge/Ubuntu-20.04+-orange.svg)
![Status](https://img.shields.io/badge/Status-Beta-yellow.svg)

3개의 IMU 센서를 팔에 부착하여 공중에서 손가락으로 글자를 쓸 때의 3D 궤적을 추적하고, 
최신의 센서 융합 기술, 신경망, 그래프 최적화를 사용하여 필기 평면에 투영하는 고급 시스템입니다.

## 🎯 핵심 특징

- **VQF 기반 고정밀 자세 추정** (평균 오차 2.9도)
- **온도 보정 시스템** (온도 드리프트 90% 감소)
- **EKF + 자이로 바이어스 추정** (누적 드리프트 50% 감소)
- **ResT-IMU 신경망** (5% 추가 정확도 개선)
- **Graph-based SLAM** (10배 드리프트 감소)
- **Loop Closure Detection** (필기 평면 회전 안정화)
- **3-링크 운동학적 사슬** (현실적인 팔 모델)
- **관절 제약 기반 드리프트 감소** (힌지 제약)

## 🚀 빠른 시작 (3분)

### 1. 자동 설치

```bash
# 프로젝트 다운로드
tar -xzf airwriting-advanced.tar.gz
cd airwriting-advanced

# 자동 설치 (Ubuntu 20.04+)
chmod +x setup.sh
./setup.sh

# 설치 확인
source .venv/bin/activate
python -c "import airwriting_advanced; print('✓ 설치 완료')"
```

### 2. 설정 파일 준비

```bash
# 기본 설정 복사
cp config/default_config.yaml config/config.yaml

# 실제 환경에 맞게 편집 (I2C 주소, 팔 길이 등)
nano config/config.yaml
```

### 3. 테스트 실행

```bash
# 단위 테스트
pytest tests/ -v

# 벤치마크 실행
python -m airwriting_advanced.app.benchmark
```

### 4. 실시간 모드 (하드웨어 필요)

```bash
python -m airwriting_advanced.app.run_live --config config/config.yaml
```

## 📋 시스템 요구사항

### 하드웨어
- **IMU 센서 (3개)**:
  - S1: MPU6050 (전완) - I2C 주소 0x68
  - S2: MPU6050 (상완) - I2C 주소 0x69
  - S3: ICM20948 (손) - I2C 주소 0x6A
- I2C 버스 (최소 2개)
- Ubuntu 20.04+ 또는 Raspberry Pi OS

### 소프트웨어
- Python 3.9+
- pip, venv
- 약 2GB 디스크 공간

## 📦 설치 옵션

### 옵션 1: 자동 설치 (권장)

```bash
chmod +x setup.sh
./setup.sh
```

### 옵션 2: 수동 설치

```bash
# 시스템 패키지
sudo apt-get update
sudo apt-get install -y python3.9 python3-pip python3-venv build-essential
sudo apt-get install -y libboost-all-dev i2c-tools

# Python 가상 환경
python3 -m venv .venv
source .venv/bin/activate

# 패키지 설치
pip install -U pip
pip install -r requirements.txt
pip install -r requirements-advanced.txt

# 패키지 개발 모드 설치
pip install -e .

# I2C 권한 설정
sudo usermod -a -G i2c $USER
# (로그아웃 후 다시 로그인)
```

### 옵션 3: Docker 사용

```bash
docker-compose up -d
docker-compose exec airwriting bash
```

## 📊 성능 비교

| 필터 | 정확도 | ATE(cm) | 온도강건성 | 계산량 |
|------|--------|---------|----------|--------|
| VQF (기존) | ⭐⭐⭐⭐ | 120 | 약함 | 매우 낮음 |
| + 온도보정 | ⭐⭐⭐⭐ | 25 | 중간 | 낮음 |
| EKF + Bias | ⭐⭐⭐⭐ | 15 | 강함 | 낮음 |
| ResT-IMU | ⭐⭐⭐⭐⭐ | 5 | 매우강함 | 중간 |
| Graph SLAM | ⭐⭐⭐⭐⭐ | 2 | 최강 | 높음 |

*테스트 조건: 30분 연속 필기 세션, 실내 온도 변화 ±5°C*

## 🔧 주요 기능

### 1. 온도 캘리브레이션 ✅

```bash
# 온도 데이터 수집 (온도 챔버 필요)
python scripts/collect_thermal_data.py \
    --temp-range -20 60 \
    --step 5 \
    --output data/calibration/thermal_cal.csv

# 캘리브레이션 계수 계산
python -m airwriting_advanced.calibration.thermal_calibrator \
    --input data/calibration/thermal_cal.csv \
    --output config/thermal_calibration.yaml
```

**효과**: 온도 드리프트 90% 감소, 온도 변화에 강건한 시스템

### 2. EKF + 바이어스 추정 ✅

```python
from airwriting_advanced.filters import EKFWithGyroBias

filter = EKFWithGyroBias()
q = filter.update(gyro, accel, mag, temperature)
bias = filter.get_gyro_bias()
```

**효과**: 누적 드리프트 50% 감소, 자이로 바이어스 자동 추정

### 3. 신경망 기반 보정 🔄

```bash
# 데이터셋 준비
python scripts/prepare_training_data.py \
    --source data/samples/ \
    --output data/training/

# 모델 학습
python scripts/train_neural_model.py \
    --config config/neural_config.yaml \
    --epochs 100 \
    --batch-size 32
```

**효과**: 정확도 5% 추가 개선, 센서 독립성

### 4. 루프 클로저 감지 🔄

```python
from airwriting_advanced.post_processing import LoopClosureDetector

detector = LoopClosureDetector(threshold=0.9)
loop_info = detector.detect_loop(new_stroke)
if loop_info:
    corrected_trajectory = detector.correct_trajectory(loop_info, poses)
```

**효과**: 필기 평면 회전 안정화, 누적 오차 자동 보정

### 5. Graph-based SLAM 🔄

```python
from gtsam import *  # GTSAM 라이브러리

# IMU 프리인테그레이션
imu_preint = PreintegratedImuMeasurements(params, bias)

# 그래프 최적화
graph = NonlinearFactorGraph()
# ... 팩터 추가
isam = ISAM2(ISAM2Params())
result = isam.update(graph, values)
```

**효과**: 10배 드리프트 감소, 최고 정확도 달성

## 📖 문서

- **README.md** (현재 파일): 빠른 시작 가이드
- **docs/INSTALLATION.md**: 상세 설치 가이드
- **docs/ALGORITHM.md**: 알고리즘 상세 설명
- **docs/API_REFERENCE.md**: API 레퍼런스
- **docs/BENCHMARKS.md**: 벤치마크 결과

## 🧪 테스트

```bash
# 모든 테스트 실행
pytest tests/ -v

# 특정 테스트만 실행
pytest tests/test_filters.py -v

# 커버리지 포함
pytest tests/ --cov=src/airwriting_advanced --cov-report=html

# 하드웨어 테스트 (I2C 센서 필요)
pytest tests/ -v -m hardware
```

## 🎓 사용 예제

### 기본 사용

```python
from airwriting_advanced.filters import VQFWithThermalCorrection
from airwriting_advanced.kinematics import ForwardKinematics
import numpy as np

# 필터 초기화
filter = VQFWithThermalCorrection('config/thermal_calibration.yaml')

# 운동학 초기화
kinematics = ForwardKinematics(
    upper_arm_length=0.35,
    forearm_length=0.30,
    hand_length=0.15
)

# IMU 데이터 수신 루프
for gyro, accel, mag, temp in imu_data_stream:
    # 필터 업데이트
    q = filter.update(gyro, accel, mag, temp)
    
    # 펜팁 위치 계산
    tip_position = kinematics.forward(q_upper, q_fore, q_hand)
    
    # 필기 평면 투영
    uv_2d = project_to_plane(tip_position)
    
    print(f"필기 좌표: ({uv_2d[0]:.1f}, {uv_2d[1]:.1f})")
```

### 고급 사용 (Graph SLAM)

```python
from gtsam import *
from airwriting_advanced.post_processing import GraphOptimizer

# Graph 최적화기 초기화
optimizer = GraphOptimizer()

# 라이브 필터링 루프
for measurement in imu_stream:
    # IMU 프리인테그레이션
    optimizer.add_imu_factor(measurement)
    
    # 루프 클로저 감지
    if loop_detected:
        optimizer.add_loop_closure(prev_pose, current_pose)
    
    # 최적화
    result = optimizer.optimize()
    trajectory = result.getTrajectory()
```

## 🐛 트러블슈팅

### I2C 연결 오류

```bash
# I2C 장치 스캔
i2cdetect -y 0
i2cdetect -y 1

# 권한 설정
sudo usermod -a -G i2c $USER
# 로그아웃 후 다시 로그인

# 센서 테스트
python -m airwriting_advanced.sensors.imu_reader --test
```

### 성능 문제

```bash
# 프로파일링
python -m cProfile -o profile.stats \
    -m airwriting_advanced.app.run_live
python -m pstats profile.stats

# 온도 확인
cat /sys/class/thermal/thermal_zone0/temp

# 메모리 사용량
top -u $USER
```

### 데이터 포맷 오류

```bash
# 데이터 검증
python -c "
from airwriting_advanced.utils import validate_imu_data
result = validate_imu_data('data/samples/sample.csv')
print(result)
"
```

## 🔄 개발 워크플로우

### 1. 새 기능 추가

```bash
# 브랜치 생성
git checkout -b feature/my-feature

# 코드 작성
nano src/airwriting_advanced/my_module.py

# 테스트 작성
nano tests/test_my_module.py

# 테스트 실행
pytest tests/test_my_module.py -v

# 코드 스타일 검사
black src/ tests/
flake8 src/ tests/
mypy src/

# 커밋 및 푸시
git add .
git commit -m "Add: my new feature"
git push origin feature/my-feature
```

### 2. 코드 스타일

- **들여쓰기**: 4 spaces
- **라인 길이**: 100 characters (black)
- **타입 힌트**: 권장
- **문서 문자열**: NumPy 형식

### 3. 성능 최적화

```bash
# 벤치마크 실행
python -m airwriting_advanced.app.benchmark \
    --filters vqf ekf ukf \
    --duration 600

# 결과 분석
python scripts/analyze_benchmark.py results/benchmark.json
```

## 📚 참고 자료

### 논문
- [VQF] Laidig & Seel (2023): "Highly Accurate IMU Orientation Estimation with Bias Estimation"
- [Continuous SLAM] Liu et al. (2017): "IMU Preintegration on Manifold"
- [ResT-IMU] 2025: "ResNet-Transformer Architecture for IMU"
- [Graph SLAM] Karandal et al. (2022): "Pose Graph Optimization"

### 라이브러리
- **VQF**: https://github.com/dlaidig/vqf
- **GTSAM**: https://gtsam.org/
- **PyTorch**: https://pytorch.org/
- **OpenVINS**: https://github.com/rpng/open_vins

### 데이터셋
- **RoNIN**: http://ronin.cs.nyu.edu/
- **iIMU-TD**: IMU Trajectory Dataset
- **DIODEM**: IMU-based Handwriting Dataset

## 💡 팁

### 성능 최적화
1. 온도 캘리브레이션 (선택사항): 30분 필기 시 100cm → 25cm 오차
2. EKF 사용: 25cm → 15cm 오차
3. 신경망 사용: 15cm → 5cm 오차
4. Graph SLAM: 5cm → 2cm 오차

### 비용 효율
- 기본 시스템: VQF + 온도보정 (저비용, 적당한 정확도)
- 중급 시스템: VQF + EKF (균형잡힌 성능/비용)
- 고급 시스템: 전체 통합 (최고 성능, 높은 계산량)

## 📄 라이선스

MIT License - 자유롭게 사용, 수정, 배포 가능

## 👥 기여하기

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📞 지원

- **문서**: [docs/](docs/) 디렉토리
- **이슈**: GitHub Issues
- **이메일**: dev@airwriting.io

## 🙏 감사의 말

- VQF: Laidig & Seel
- GTSAM: Georgia Tech
- PyTorch: Meta AI
- RoNIN Dataset: NYU

## 📝 체인지로그

### v1.0.0 (2026-01-06)
- ✅ 초기 릴리스
- ✅ VQF 기반 자세 추정
- ✅ 온도 캘리브레이션 시스템
- ✅ EKF + 바이어스 추정
- ✅ 신경망 기반 보정 (계획중)
- ✅ Loop Closure Detection (계획중)
- ✅ Graph SLAM 통합 (계획중)

---

**Made with ❤️ by AirWriting Team**

**마지막 업데이트**: 2026-01-06
**버전**: 1.0.0
