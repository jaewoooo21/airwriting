@echo off
chcp 65001 >nul
echo ==============================================
echo   방화벽 포트 18800 개방 (관리자 권한 요청)
echo ==============================================
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo 관리자 권한이 필요합니다. 팝업창에서 '예'를 눌러주세요...
    powershell -Command "Start-Process '%~dpnx0' -Verb RunAs"
    exit /b
)
echo.
echo [1/1] 포트 18800 (TCP) 인바운드 규칙을 추가합니다...
netsh advfirewall firewall add rule name="AirWriting Action (Port 18800)" dir=in action=allow protocol=TCP localport=18800 >nul
echo.
echo 완료되었습니다! 이제 핸드폰에서 새로고침을 해보세요.
echo 창을 닫으셔도 됩니다.
pause
