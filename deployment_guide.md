# AirWriting Dashboard 배포 가이드 (Cloud Deployment Guide)

이 문서는 완성된 AirWriting 대시보드와 팀 댓글 게시판을 다른 팀원들이 볼 수 있도록 **무료 호스팅 서비스(Render.com 등)**에 배포하는 방법을 설명합니다.

## 🚀 사전 준비 (GitHub)
가장 쉽고 자동화된 배포를 위해, 작성된 코드를 GitHub 리포지토리에 업로드해야 합니다.

1. `airwriting_imu_only` 폴더 내, 특히 `web_app` 폴더의 코드(`index.html`, `style.css`, `app.js`, `app.py`, `requirements.txt`)가 빠짐없이 커밋되어 있는지 확인합니다.
2. 만약 GitHub에 전체 프로젝트가 올라가 있다면, 그 리포지토리를 그대로 사용하면 됩니다.

---

## 옵션 1: Render.com 을 이용한 무료 웹 서비스 배포 (추천)
Render는 GitHub 연동 시 버튼 몇 번으로 파이썬 웹서비스를 매우 쉽게 호스팅해 주는 서비스입니다.

### 배포 순서
1. [Render.com](https://render.com) 에 접속하여 회원가입(GitHub 계정 연동)을 합니다.
2. Dashboard에서 **"New"** -> **"Web Service"** 를 클릭합니다.
3. 배포할 코드가 담긴 본인의 **GitHub Repository**를 선택합니다.
4. 설정 화면에서 다음 항목들을 알맞게 입력합니다:
   - **Name**: `airwriting-dashboard` (원하는 이름)
   - **Environment**: `Python 3`
   - **Region**: (기본값)
   - **Branch**: `main` (또는 코드가 있는 브랜치 명)
   - **Root Directory**: `web_app` (중요! `app.py`와 `requirements.txt`가 있는 폴더명입니다)
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app` (Gunicorn을 사용해 배포 환경에서 Flask를 실행합니다)
5. **"Free"** 플랜(무료)을 선택하고 **Create Web Service** 버튼을 누릅니다.
6. 몇 분의 빌드 과정이 끝나면, Render에서 제공하는 무료 도메인 `https://airwriting-dashboard.onrender.com` 과 같은 주소가 생성됩니다.
7. 이 주소를 방문하여 대시보드와 댓글 창이 잘 열리는지 확인한 뒤, 팀원들에게 공유합니다!

> 💡 **참고:** SQLite 파일(`comments.db`)은 Render 무료 티어의 디스크 휘발성으로 인해 서버가 재시작되면(며칠 단위로) 댓글 로그가 초기화될 수 있습니다. 만약 영구 저장이 매우 중요하다면 PythonAnywhere를 사용하거나 Render에 결제를 통해 디스크 마운트를 해야 합니다. (졸작 피드백 용도라면 PythonAnywhere를 추천합니다!)

---

## 옵션 2: PythonAnywhere 를 이용한 무료 배포 (영구 저장 특화)
데이터베이스 초기화가 걱정된다면 PythonAnywhere를 사용하세요. 약간 더 설정이 필요하지만 무료 계정에서도 SQLite 파일이 영구 유지됩니다.

### 배포 순서
1. [PythonAnywhere.com](https://www.pythonanywhere.com) 에 가입합니다.
2. Dashboard에서 **"Web"** 탭 -> **"Add a new web app"** 버튼을 클릭합니다.
3. 도메인 지정(기본 제공되는 `유저명.pythonanywhere.com` 사용) 후 프레임워크 선택 창에서 **"Flask"**를 선택하고, 파이썬 버전은 **"Python 3.10"**(또는 가장 최신 버전)을 선택합니다.
4. 기본 Flask 앱 경로가 나오면 그대로 저장합니다.
5. 상단의 **"Files"** 탭으로 이동하여, `mysite` (또는 기본으로 생성된 앱 폴더) 안에 들어갑니다.
6. PC에 있는 `web_app` 폴더 안의 5가지 핵심 파일 (`app.py`, `index.html`, `app.js`, `style.css`, `requirements.txt`)을 모두 여기로 **업로드(Upload)** 합니다.
   - ⚠️ 업로드 시 기존 화면에 있던 `flask_app.py` 같은 파일이 있다면 삭제하고, `app.py`를 메인으로 쓰도록 "Web" 탭의 WSGI 설정 파일을 수정해야 할 수도 있습니다. 
   - (초보자라면 업로드할 때 파일명을 `flask_app.py`로 바꿔치기 하는 것도 좋은 팁입니다!)
7. **"Consoles"** 탭에서 **Bash** 콘솔을 열고 `pip install -r requirements.txt` (Flask 설치) 를 입력/실행합니다.
8. **"Web"** 탭 하단의 초록색 🔄 **Reload** 버튼을 누릅니다.
9. 제공된 본인만의 도메인 이름(예: `https://yourname.pythonanywhere.com`)으로 접속해 팀원들과 공유합니다!

---

### 로컬 테스트 (자신의 PC에서 백엔드 켜기)
지금처럼 로컬 화면을 확인할 때는 터미널을 열고 다음 명령어를 칩니다.
```bash
cd web_app
python app.py
```
이후 웹 브라우저에서 `http://localhost:5000` 에 접속하면 프론트엔드와 백엔드가 결합된 가장 완벽한 상태의 페이지를 볼 수 있습니다. (기존에 켜두신 `python digital_twin.py` 등과 같이 켜두면 됩니다!)
