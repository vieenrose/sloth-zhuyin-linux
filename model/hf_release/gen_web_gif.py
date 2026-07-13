#!/usr/bin/env python3
"""Headless-Chrome + CDP web-demo GIF generator (self-contained on this host).
Drives the live 25M-ternary web demo, types 我用claude寫程式 via injected keydown
events (the demo's document keydown handler), screenshots each step, assembles a GIF."""
import json, subprocess, time, urllib.request, base64, os, sys, shutil
import websocket  # websocket-client

URL = "https://luigi-slothing-web.static.hf.space/index.html?v=25m"
OUT = "/tmp/webframes"
KEYSTREAM = [("ji3", "我"), ("m/4", "我用"), ("claude", "我用claude"),
             ("vu,3t/6g4", "我用claude寫程式")]
shutil.rmtree(OUT, ignore_errors=True); os.makedirs(OUT)

chrome = shutil.which("google-chrome") or shutil.which("google-chrome-stable")
prof = "/tmp/cdp_prof"; shutil.rmtree(prof, ignore_errors=True)
proc = subprocess.Popen([chrome, "--headless=new", "--disable-gpu", "--no-sandbox",
    "--remote-debugging-port=9223", "--remote-allow-origins=*", "--user-data-dir=" + prof,
    "--window-size=520,900", "--force-device-scale-factor=1.5", "--hide-scrollbars",
    "--disable-features=Translate", "about:blank"],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def wait_devtools():
    for _ in range(40):
        try:
            tabs = json.load(urllib.request.urlopen("http://localhost:9223/json"))
            for t in tabs:
                if t.get("type") == "page" and t.get("webSocketDebuggerUrl"):
                    return t["webSocketDebuggerUrl"]
        except Exception: pass
        time.sleep(0.5)
    raise RuntimeError("devtools not up")

ws = websocket.create_connection(wait_devtools(), max_size=None)
ws.settimeout(30)
_id = [0]
def cmd(method, params=None):
    _id[0] += 1
    ws.send(json.dumps({"id": _id[0], "method": method, "params": params or {}}))
    import time as _t; t0=_t.time()
    while _t.time()-t0 < 35:
        try:
            m = json.loads(ws.recv())
        except Exception:
            return {}
        if m.get("id") == _id[0]:
            return m.get("result", {})
    return {}

cmd("Page.enable"); cmd("Runtime.enable")
cmd("Page.navigate", {"url": URL})

def evaltxt(expr):
    r = cmd("Runtime.evaluate", {"expression": expr, "returnByValue": True})
    return (r.get("result") or {}).get("value")

# wait for model ready (hint text flips to the ready message)
ready = False
for _ in range(50):
    h = evaltxt("(document.getElementById('hint')||{}).textContent||''") or ""
    if "自動辨識" in h:
        ready = True; break
    time.sleep(1)
print("model ready:", ready)
time.sleep(1)

nframe = [0]
def shot(tag):
    r = cmd("Page.captureScreenshot", {"format": "png"})
    data = r.get("data")
    if data:
        open(f"{OUT}/f{nframe[0]:02d}_{tag}.png", "wb").write(base64.b64decode(data))
        nframe[0] += 1

def key(ch):
    # the demo reads e.key on a document keydown listener
    cmd("Input.dispatchKeyEvent", {"type": "keyDown", "key": ch, "text": ch})
    cmd("Input.dispatchKeyEvent", {"type": "keyUp", "key": ch, "text": ch})
    time.sleep(0.05)

shot("empty")
for chunk, label in KEYSTREAM:
    for ch in chunk:
        key(ch)
    time.sleep(0.5)
    got = evaltxt("(document.getElementById('pre')||{}).textContent||''")
    print(f"after {chunk!r}: composing={got!r} (want {label!r})")
    shot("t" + str(nframe[0]))
# commit
evaltxt("var b=document.getElementById('commit'); if(b) b.click(); true")
time.sleep(0.7); shot("commit")
out_val = evaltxt("(document.getElementById('out')||{}).value||''")
print("committed output:", out_val)

ws.close(); proc.terminate()
print("frames:", sorted(os.listdir(OUT)))
