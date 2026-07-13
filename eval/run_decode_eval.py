#!/usr/bin/env python3
"""Stage A probe: libchewing-free LLM decoding vs the libchewing baseline.

Feeds each testset case's bopomofo syllables straight to slothd's decode
mode ({"syllables": [...]}) -- no libchewing candidates involved; legality
comes from model/phonetic_table.tsv. Scores three decoders on the same cases:

  chewing   : libchewing's own buffer (via eval/harvest), the incumbent
  decode    : LLM constrained to phonetically-legal chars, tones kept
  toneless  : same but tones stripped -- the future tone-free typing mode

The daemon must be started with -t:
  slothd -m model.gguf -s $SOCK -t model/phonetic_table.tsv

Usage: python3 eval/run_decode_eval.py [--socket PATH] [--limit N] [--verbose]
"""
import argparse
import json
import os
import socket
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TONES = "ˊˇˋ˙"

# Same layout map as run_eval.py, for the chewing-baseline harvest only.
KEYMAP = {
    "ㄅ": "1", "ㄆ": "q", "ㄇ": "a", "ㄈ": "z", "ㄉ": "2", "ㄊ": "w", "ㄋ": "s",
    "ㄌ": "x", "ㄍ": "e", "ㄎ": "d", "ㄏ": "c", "ㄐ": "r", "ㄑ": "f", "ㄒ": "v",
    "ㄓ": "5", "ㄔ": "t", "ㄕ": "g", "ㄖ": "b", "ㄗ": "y", "ㄘ": "h", "ㄙ": "n",
    "ㄧ": "u", "ㄨ": "j", "ㄩ": "m", "ㄚ": "8", "ㄛ": "i", "ㄜ": "k", "ㄝ": ",",
    "ㄞ": "9", "ㄟ": "o", "ㄠ": "l", "ㄡ": ".", "ㄢ": "0", "ㄣ": "p", "ㄤ": ";",
    "ㄥ": "/", "ㄦ": "-", "ˊ": "6", "ˇ": "3", "ˋ": "4", "˙": "7",
}


def bopomofo_to_keys(bopomofo):
    keys = []
    for syl in bopomofo.split():
        keys += [KEYMAP[c] for c in syl]
        if syl[-1] not in TONES:
            keys.append(" ")
    return "".join(keys)


def query(sock_path, syllables, context):
    req = {"syllables": syllables, "n": 1}
    if context:
        req["context"] = context
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(60)
    s.connect(sock_path)
    t0 = time.time()
    s.sendall((json.dumps(req) + "\n").encode())
    s.shutdown(socket.SHUT_WR)
    resp = b""
    while True:
        chunk = s.recv(65536)
        if not chunk:
            break
        resp += chunk
    s.close()
    return json.loads(resp), time.time() - t0


def score(name, results, latencies):
    n = len(results)
    sent = sum(1 for got, exp in results if got == exp)
    tot = ok = 0
    for got, exp in results:
        ok += sum(1 for a, b in zip(got, exp) if a == b)
        tot += max(len(got), len(exp))
    line = (f"{name:<9} sentence {sent}/{n} ({100*sent/n:.0f}%), "
            f"char {ok}/{tot} ({100*ok/tot:.1f}%)")
    if latencies:
        lat = sorted(latencies)
        line += (f", median {lat[len(lat)//2]*1000:.0f}ms, "
                 f"p95 {lat[int(len(lat)*0.95)]*1000:.0f}ms")
    print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--socket",
                    default=os.environ.get("SLOTHD_SOCKET")
                    or os.path.join(os.environ.get("XDG_RUNTIME_DIR", "/tmp"),
                                    "slothd.sock"))
    ap.add_argument("--limit", type=int, default=0, help="0 = all cases")
    ap.add_argument("--no-toneless", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    harvest_bin = os.path.join(HERE, "harvest")
    if not os.path.exists(harvest_bin):
        sys.exit("build eval/harvest first")

    cases = []
    with open(os.path.join(HERE, "testset.tsv"), encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            p = line.split("\t")
            cases.append((p[0], p[1], p[2] if len(p) > 2 else ""))
    if args.limit:
        cases = cases[:args.limit]

    chew, dec, tless = [], [], []
    lat_dec, lat_tless = [], []
    skipped = 0
    for bopomofo, expected, context in cases:
        syls = bopomofo.split()
        # chewing baseline
        out = subprocess.run([harvest_bin, bopomofo_to_keys(bopomofo)],
                             capture_output=True, text=True)
        try:
            buffer = json.loads(out.stdout)["buffer"]
        except Exception:
            skipped += 1
            continue

        resp, dt = query(args.socket, syls, context)
        if "error" in resp:
            print(f"  decode error on {expected}: {resp['error']}",
                  file=sys.stderr)
            skipped += 1
            continue
        got = resp["sentences"][0] if resp.get("sentences") else ""
        chew.append((buffer, expected))
        dec.append((got, expected))
        lat_dec.append(dt)

        if not args.no_toneless:
            tl = ["".join(c for c in s if c not in TONES) for s in syls]
            resp, dt = query(args.socket, tl, context)
            gt = resp["sentences"][0] if resp.get("sentences") else ""
            tless.append((gt, expected))
            lat_tless.append(dt)
        else:
            gt = None

        if args.verbose or got != expected:
            print(f"[{'OK ' if got == expected else 'MISS'}] expect={expected} "
                  f"chewing={buffer} decode={got}"
                  + (f" toneless={gt}" if gt is not None else ""))

    print(f"\n=== decode probe: {len(dec)} cases, {skipped} skipped ===")
    score("chewing", chew, [])
    score("decode", dec, lat_dec)
    if tless:
        score("toneless", tless, lat_tless)


if __name__ == "__main__":
    main()
