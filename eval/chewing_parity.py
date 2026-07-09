#!/usr/bin/env python3
"""Chewing-parity gate: validate SlothLM against real libchewing behaviour.

STANDING RULE: every model / decode change is validated against libchewing,
not eyeballed. libchewing is the reference IME; SlothLM must not regress
against it and should win overall.

For each testset case this compares, on the SAME zhuyin input:
  * libchewing's own decode (via eval/harvest, the exact engine harvest)
  * SlothLM's decode  (slothingd, decode mode)
against the expected sentence, and reports:
  * sentence accuracy for each (must have: LLM >= chewing)
  * agreement rate (how often they already produce the same thing)
  * divergences: where they differ, who is right (the interesting cases)

Run:
  packaging/run-slothingd.sh   # (with the model under test)
  python3 eval/chewing_parity.py [--socket PATH] [--show N]
Exit code is non-zero if the LLM regresses vs chewing on sentence accuracy,
so it can gate CI / a model swap.
"""
import argparse
import json
import os
import socket
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TONES = "ˊˇˋ˙"
KEYMAP = {
    "ㄅ": "1", "ㄆ": "q", "ㄇ": "a", "ㄈ": "z", "ㄉ": "2", "ㄊ": "w", "ㄋ": "s",
    "ㄌ": "x", "ㄍ": "e", "ㄎ": "d", "ㄏ": "c", "ㄐ": "r", "ㄑ": "f", "ㄒ": "v",
    "ㄓ": "5", "ㄔ": "t", "ㄕ": "g", "ㄖ": "b", "ㄗ": "y", "ㄘ": "h", "ㄙ": "n",
    "ㄧ": "u", "ㄨ": "j", "ㄩ": "m", "ㄚ": "8", "ㄛ": "i", "ㄜ": "k", "ㄝ": ",",
    "ㄞ": "9", "ㄟ": "o", "ㄠ": "l", "ㄡ": ".", "ㄢ": "0", "ㄣ": "p", "ㄤ": ";",
    "ㄥ": "/", "ㄦ": "-", "ˊ": "6", "ˇ": "3", "ˋ": "4", "˙": "7",
}


def keys(bopomofo):
    k = []
    for syl in bopomofo.split():
        k += [KEYMAP[c] for c in syl]
        if syl[-1] not in TONES:
            k.append(" ")
    return "".join(k)


def slothlm(sock_path, syllables):
    req = {"syllables": syllables, "n": 1}
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(30)
    s.connect(sock_path)
    s.sendall((json.dumps(req) + "\n").encode())
    s.shutdown(socket.SHUT_WR)
    resp = b""
    while True:
        c = s.recv(65536)
        if not c:
            break
        resp += c
    s.close()
    d = json.loads(resp)
    return (d.get("sentences") or [""])[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--socket",
                    default=os.environ.get("SLOTHINGD_SOCKET")
                    or os.path.join(os.environ.get("XDG_RUNTIME_DIR", "/tmp"),
                                    "slothingd.sock"))
    ap.add_argument("--show", type=int, default=12,
                    help="how many divergences to print")
    args = ap.parse_args()
    harvest = os.path.join(HERE, "harvest")
    if not os.path.exists(harvest):
        sys.exit("build eval/harvest first")

    n = ch_ok = lm_ok = agree = both_wrong = 0
    lm_wins, ch_wins, divergences = [], [], []
    for line in open(os.path.join(HERE, "testset.tsv"), encoding="utf-8"):
        line = line.rstrip("\n")
        if not line or line.startswith("#"):
            continue
        p = line.split("\t")
        bopomofo, expected = p[0], p[1]
        h = json.loads(subprocess.run([harvest, keys(bopomofo)],
                                      capture_output=True, text=True).stdout)
        chew = h["buffer"]
        lm = slothlm(args.socket, bopomofo.split())
        n += 1
        ch_ok += chew == expected
        lm_ok += lm == expected
        if chew == lm:
            agree += 1
        else:
            divergences.append((expected, chew, lm))
            if lm == expected and chew != expected:
                lm_wins.append((expected, chew, lm))
            elif chew == expected and lm != expected:
                ch_wins.append((expected, chew, lm))
            elif chew != expected and lm != expected:
                both_wrong += 1

    print(f"=== chewing-parity: {n} cases ===")
    print(f"libchewing sentence: {ch_ok}/{n} ({100*ch_ok/n:.0f}%)")
    print(f"SlothLM    sentence: {lm_ok}/{n} ({100*lm_ok/n:.0f}%)  "
          f"[{'PASS' if lm_ok >= ch_ok else 'REGRESSION'}]")
    print(f"agree on {agree}/{n} ({100*agree/n:.0f}%); "
          f"diverge on {len(divergences)}")
    print(f"  of divergences: SlothLM right {len(lm_wins)}, "
          f"chewing right {len(ch_wins)}, both wrong {both_wrong}")
    if ch_wins:
        print(f"\n-- cases where chewing beats SlothLM (fix these) --")
        for exp, chew, lm in ch_wins[:args.show]:
            print(f"  want {exp}  chewing={chew}  SlothLM={lm}")
    if lm_wins:
        print(f"\n-- cases where SlothLM beats chewing --")
        for exp, chew, lm in lm_wins[:args.show]:
            print(f"  want {exp}  chewing={chew}  SlothLM={lm}")

    sys.exit(0 if lm_ok >= ch_ok else 1)


if __name__ == "__main__":
    main()
