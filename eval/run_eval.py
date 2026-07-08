#!/usr/bin/env python3
"""Slothing evaluation harness.

Reads eval/testset.tsv (bopomofo \t expected [\t context]), converts bopomofo
to default-layout (Dachen) key sequences, harvests libchewing candidates via
./harvest (same logic as the engine), queries the slothingd socket, and scores
the LLM's top-1 sentence against the expected text.

Usage: python3 eval/run_eval.py [--socket PATH] [--n 4]
"""
import argparse
import json
import os
import socket
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

# Standard (Dachen) zhuyin layout
KEYMAP = {
    "ㄅ": "1", "ㄆ": "q", "ㄇ": "a", "ㄈ": "z",
    "ㄉ": "2", "ㄊ": "w", "ㄋ": "s", "ㄌ": "x",
    "ㄍ": "e", "ㄎ": "d", "ㄏ": "c",
    "ㄐ": "r", "ㄑ": "f", "ㄒ": "v",
    "ㄓ": "5", "ㄔ": "t", "ㄕ": "g", "ㄖ": "b",
    "ㄗ": "y", "ㄘ": "h", "ㄙ": "n",
    "ㄧ": "u", "ㄨ": "j", "ㄩ": "m",
    "ㄚ": "8", "ㄛ": "i", "ㄜ": "k", "ㄝ": ",",
    "ㄞ": "9", "ㄟ": "o", "ㄠ": "l", "ㄡ": ".",
    "ㄢ": "0", "ㄣ": "p", "ㄤ": ";", "ㄥ": "/", "ㄦ": "-",
    "ˊ": "6", "ˇ": "3", "ˋ": "4", "˙": "7",
}
TONES = "ˊˇˋ˙"


def bopomofo_to_keys(bopomofo: str) -> str:
    keys = []
    for syllable in bopomofo.split():
        for ch in syllable:
            if ch not in KEYMAP:
                raise ValueError(f"unmapped bopomofo symbol {ch!r} in {syllable!r}")
            keys.append(KEYMAP[ch])
        if syllable[-1] not in TONES:
            keys.append(" ")  # tone 1
    return "".join(keys)


def query_daemon(sock_path, positions, n, context):
    req = {"positions": positions, "n": n}
    if context:
        req["context"] = context
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(20)
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
    dt = time.time() - t0
    s.close()
    return json.loads(resp), dt


def char_accuracy(got: str, expected: str):
    g, e = list(got), list(expected)
    correct = sum(1 for a, b in zip(g, e) if a == b)
    return correct, max(len(g), len(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--socket",
                    default=os.environ.get("SLOTHINGD_SOCKET")
                    or os.path.join(os.environ.get("XDG_RUNTIME_DIR", "/tmp"),
                                    "slothingd.sock"))
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    harvest_bin = os.path.join(HERE, "harvest")
    if not os.path.exists(harvest_bin):
        sys.exit("build eval/harvest first: "
                 "gcc eval/harvest.c -o eval/harvest $(pkg-config --cflags --libs chewing)")

    cases = []
    with open(os.path.join(HERE, "testset.tsv"), encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            cases.append((parts[0], parts[1], parts[2] if len(parts) > 2 else ""))

    tot_chars = ok_chars = 0
    llm_sent_ok = chewing_sent_ok = 0
    llm_recall = 0  # expected is somewhere in the candidate list (pick-UX metric)
    chewing_ok_chars = 0
    latencies = []
    failures = []

    for bopomofo, expected, context in cases:
        keys = bopomofo_to_keys(bopomofo)
        out = subprocess.run([harvest_bin, keys], capture_output=True, text=True)
        if out.returncode != 0:
            failures.append((expected, f"harvest failed: {out.stderr.strip()}"))
            continue
        h = json.loads(out.stdout)
        buffer, positions = h["buffer"], h["positions"]
        if len(buffer) != len(expected):
            failures.append((expected, f"buffer {buffer!r} length mismatch "
                                       f"(check bopomofo)"))
            continue

        resp, dt = query_daemon(args.socket, positions, args.n, context)
        latencies.append(dt)
        sentences = resp.get("sentences", [])
        top1 = sentences[0] if sentences else buffer  # daemon empty -> chewing

        c_ok, c_tot = char_accuracy(top1, expected)
        ok_chars += c_ok
        tot_chars += c_tot
        llm_sent_ok += int(top1 == expected)
        # Pick-from-list UX: is the right sentence reachable at all? The engine
        # also always offers chewing's own sentence, so count it as in-list.
        llm_recall += int(expected in sentences or expected == buffer)
        b_ok, _ = char_accuracy(buffer, expected)
        chewing_ok_chars += b_ok
        chewing_sent_ok += int(buffer == expected)

        marker = "OK " if top1 == expected else "MISS"
        if args.verbose or top1 != expected:
            print(f"[{marker}] expect={expected} chewing={buffer} "
                  f"llm={top1} ({dt*1000:.0f}ms)")

    n = len(cases) - len(failures)
    print("\n=== summary ===")
    print(f"cases: {n} scored, {len(failures)} skipped")
    for exp, why in failures:
        print(f"  skipped {exp}: {why}")
    if n:
        print(f"chewing baseline: sentence {chewing_sent_ok}/{n} "
              f"({100*chewing_sent_ok/n:.0f}%), "
              f"char {chewing_ok_chars}/{tot_chars} "
              f"({100*chewing_ok_chars/tot_chars:.1f}%)")
        print(f"LLM top-1:        sentence {llm_sent_ok}/{n} "
              f"({100*llm_sent_ok/n:.0f}%), "
              f"char {ok_chars}/{tot_chars} ({100*ok_chars/tot_chars:.1f}%)")
        print(f"LLM recall (in list, pick-UX): {llm_recall}/{n} "
              f"({100*llm_recall/n:.0f}%)")
        lat = sorted(latencies)
        print(f"latency: median {lat[len(lat)//2]*1000:.0f}ms, "
              f"p95 {lat[int(len(lat)*0.95)]*1000:.0f}ms")


if __name__ == "__main__":
    main()
