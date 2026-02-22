import socket, json
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind(("0.0.0.0", 12346))
print("Listening on 12346 for unity packets...")
for _ in range(5):
    data, addr = sock.recvfrom(4096)
    d = json.loads(data.decode('utf-8'))
    print(f"S3fk: {d.get('S3fk')}")
    print(f"S3z:  {d.get('S3z')}")
    print(f"PEN:  {d.get('pen')}")
    print("-------------------------")
