#!/bin/bash

# setup.sh - AirWriting Advanced 자동 설치 스크립트
# Ubuntu 20.04+ 환경을 위한 완전 자동 설정

set -e  # 에러 발생시 즉시 종료

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 로깅 함수
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[⚠]${NC} $1"
}

log_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# 1. 시스템 정보 확인
log_info "="*50
log_info "AirWriting Advanced - 자동 설치 시작"
log_info "="*50

OS=$(lsb_release -d | cut -f2)
KERNEL=$(uname -r)
PYTHON=$(python3 --version)

log_info "운영 체제: $OS"
log_info "커널: $KERNEL"
log_info "Python: $PYTHON"

# 2. 필수 시스템 패키지 설치
log_info ""
log_info "1단계: 시스템 패키지 설치 중..."

# apt 업데이트
sudo apt-get update -qq

# 필수 패키지 설치
PACKAGES=(
    "python3.9"
    "python3-pip"
    "python3-venv"
    "build-essential"
    "libatlas-base-dev"
    "libjasper-dev"
    "libtiff5"
    "libjasper1"
    "libharfbuzz0b"
    "libwebp6"
    "libboost-all-dev"
    "cmake"
    "git"
    "i2c-tools"
    "python3-smbus2"
    "libopenblas-dev"
    "liblapack-dev"
    "gfortran"
)

for pkg in "${PACKAGES[@]}"; do
    if dpkg -l | grep -q "^ii  $pkg"; then
        log_success "$pkg 이미 설치됨"
    else
        log_info "$pkg 설치 중..."
        sudo apt-get install -y -qq "$pkg" || log_warning "$pkg 설치 실패 (선택사항)"
    fi
done

log_success "시스템 패키지 설치 완료"

# 3. Python 가상 환경 생성
log_info ""
log_info "2단계: Python 가상 환경 설정 중..."

if [ -d ".venv" ]; then
    log_warning "가상 환경이 이미 존재합니다. 삭제 후 재생성합니다."
    rm -rf .venv
fi

python3 -m venv .venv
log_success "가상 환경 생성 완료: .venv/"

# 4. 가상 환경 활성화 및 패키지 설치
log_info ""
log_info "3단계: Python 패키지 설치 중..."

source .venv/bin/activate

# pip 업그레이드
log_info "pip 업그레이드 중..."
pip install -q --upgrade pip setuptools wheel

# 기본 패키지 설치
log_info "기본 패키지 설치 중..."
if [ -f "requirements.txt" ]; then
    pip install -q -r requirements.txt
    log_success "기본 패키지 설치 완료"
else
    log_warning "requirements.txt를 찾을 수 없습니다"
fi

# 고급 패키지 설치 (옵션)
if [ -f "requirements-advanced.txt" ]; then
    log_info "고급 패키지 설치 중... (이 과정은 시간이 걸릴 수 있습니다)"
    pip install -q -r requirements-advanced.txt 2>/dev/null || log_warning "일부 고급 패키지 설치 실패"
    log_success "고급 패키지 설치 완료"
else
    log_warning "requirements-advanced.txt를 찾을 수 없습니다"
fi

# 개발 패키지 설치 (옵션)
if [ -f "requirements-dev.txt" ]; then
    log_info "개발 패키지 설치 중..."
    pip install -q -r requirements-dev.txt || log_warning "일부 개발 패키지 설치 실패"
    log_success "개발 패키지 설치 완료"
fi

log_success "Python 패키지 설치 완료"

# 5. 패키지 설치 (editable mode)
log_info ""
log_info "4단계: AirWriting Advanced 패키지 설치 중..."

if [ -f "pyproject.toml" ]; then
    pip install -e . -q
    log_success "AirWriting Advanced 설치 완료"
else
    log_warning "pyproject.toml를 찾을 수 없습니다"
fi

# 6. 설정 파일 준비
log_info ""
log_info "5단계: 설정 파일 준비 중..."

if [ ! -d "config" ]; then
    mkdir -p config
fi

if [ ! -f "config/config.yaml" ]; then
    if [ -f "config/default_config.yaml" ]; then
        cp config/default_config.yaml config/config.yaml
        log_success "설정 파일 생성: config/config.yaml"
    fi
fi

# 7. 데이터 디렉토리 생성
log_info ""
log_info "6단계: 데이터 디렉토리 생성 중..."

mkdir -p data/calibration
mkdir -p data/models
mkdir -p data/samples
mkdir -p data/logs
mkdir -p results

log_success "데이터 디렉토리 생성 완료"

# 8. I2C 권한 설정
log_info ""
log_info "7단계: I2C 권한 설정 중..."

if groups $USER | grep -q "\bi2c\b"; then
    log_success "사용자가 이미 i2c 그룹에 속해있습니다"
else
    sudo usermod -a -G i2c $USER
    log_warning "사용자를 i2c 그룹에 추가했습니다 (로그아웃 후 다시 로그인 필요)"
fi

# 9. 테스트 실행
log_info ""
log_info "8단계: 설치 확인 중..."

python -c "import airwriting_advanced; print('✓ AirWriting Advanced 임포트 성공')" && log_success "AirWriting Advanced 임포트 성공" || log_error "임포트 실패"

# 10. 최종 요약
log_info ""
log_info "="*50
log_success "설치 완료!"
log_info "="*50

echo ""
log_info "다음 단계:"
echo "1. 가상 환경 활성화:"
echo "   ${BLUE}source .venv/bin/activate${NC}"
echo ""
echo "2. 설정 파일 편집:"
echo "   ${BLUE}nano config/config.yaml${NC}"
echo ""
echo "3. 테스트 실행:"
echo "   ${BLUE}pytest tests/ -v${NC}"
echo ""
echo "4. 실시간 실행 (하드웨어 필요):"
echo "   ${BLUE}python -m airwriting_advanced.app.run_live --config config/config.yaml${NC}"
echo ""
echo "5. 오프라인 분석:"
echo "   ${BLUE}python -m airwriting_advanced.app.run_offline --csv data/samples/sample.csv${NC}"
echo ""
log_info "상세한 문서는 README.md와 docs/ 디렉토리를 참고하세요"
echo ""

# I2C 권한 재로그인 메시지
if ! groups $USER | grep -q "\bi2c\b"; then
    log_warning ""
    log_warning "I2C 센서를 사용하려면 다음 명령으로 로그아웃 후 다시 로그인하세요:"
    log_warning "logout 또는 sudo reboot"
    echo ""
fi

log_success "설치 스크립트 종료"
