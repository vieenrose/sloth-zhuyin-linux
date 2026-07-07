import socket, json, sys, time

TESTS = [
    [["你","妳","擬"],["好","號","毫"],["謝","邪","斜"],["謝","邪","斜"],["你","妳","擬"]],
    [["我","握","沃"],["在","再"],["吃","池","馳"],["飯","犯","範"],["的","得","地"],["時","十"],["候","後","侯"]],
]

sock_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/slothingd.sock"

for positions in TESTS:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect(sock_path)
    req = json.dumps({"positions": positions}) + "\n"
    t0 = time.time()
    s.sendall(req.encode())
    s.shutdown(socket.SHUT_WR)
    resp = b""
    while True:
        chunk = s.recv(4096)
        if not chunk:
            break
        resp += chunk
    dt = time.time() - t0
    print(f"{resp.decode().strip()}   ({dt*1000:.0f}ms)")
