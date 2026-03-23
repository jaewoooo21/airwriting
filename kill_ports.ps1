$ports = @(12345, 12346, 12347, 12348, 18765, 18800, 5000)
foreach ($port in $ports) {
    echo "Checking port $port..."
    $process = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique
    if ($process) {
        foreach ($p in $process) {
            echo "Stopping process $p on port $port"
            Stop-Process -Id $p -Force -ErrorAction SilentlyContinue
        }
    }
    $udp = Get-NetUDPEndpoint -LocalPort $port -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique
    if ($udp) {
        foreach ($p in $udp) {
            echo "Stopping UDP process $p on port $port"
            Stop-Process -Id $p -Force -ErrorAction SilentlyContinue
        }
    }
}
echo "Cleanup complete."
