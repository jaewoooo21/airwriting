import socket

print("Starting UDP Sniffer on port 12345...")
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind(("0.0.0.0", 12345))

count = 0
while count < 3:
    data, addr = sock.recvfrom(4096)
    print(f"[{count}] Received {len(data)} bytes from {addr}")
    if len(data) >= 78:
        print(f"  Header: {hex(data[0])}, Footer V2: {hex(data[79]) if len(data)>79 else 'N/A'}")
        if len(data) >= 92:
            print(f"  Footer V3: {hex(data[91])}")
    count += 1
print("Done.")
