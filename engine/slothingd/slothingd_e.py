#!/usr/bin/env python3
"""slothingd-e: SlothLM-E (ONNX encoder) decode daemon for the fcitx5 engine.

Drop-in replacement for the llama.cpp slothingd, speaking the same socket
protocol as engine/fcitx5-chewing/src/eim.cpp:

  request:  {"syllables": ["ㄋㄧˇ","ㄏㄠˇ"], "n": 3, "context": "..."}\n
  response: {"sentences": ["你好", ...]}   (best first, per-position legal)

Decode is one encoder forward + per-position argmax over each syllable's
phonetically legal characters (model/phonetic_table.tsv). Alternates for n>1
flip the lowest-margin positions to their runner-up characters. `context` is
accepted but unused (the encoder is context-free by design).

  python3 engine/slothingd/slothingd_e.py \
      --model model/slothe_4m_onnx --table model/phonetic_table.tsv \
      [-s /path/to/socket]
"""
import argparse, json, os, socket, sys
import numpy as np
import onnxruntime as ort

TONES = "ˊˇˋ˙"


def default_socket_path():
    if os.environ.get("SLOTHINGD_SOCKET"):
        return os.environ["SLOTHINGD_SOCKET"]
    if os.environ.get("XDG_RUNTIME_DIR"):
        return os.path.join(os.environ["XDG_RUNTIME_DIR"], "slothingd.sock")
    return "/tmp/slothingd.sock"


class Decoder:
    def __init__(self, model_dir, table_path):
        onnx = os.path.join(model_dir, "model_quantized.onnx")
        if not os.path.exists(onnx):
            onnx = os.path.join(model_dir, "model.onnx")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2      # keystroke latency, not throughput
        self.sess = ort.InferenceSession(onnx, opts,
                                         providers=["CPUExecutionProvider"])
        self.syl_vocab = json.load(open(os.path.join(model_dir, "syl_vocab.json"),
                                        encoding="utf-8"))
        # tokenizer-free serving: char ids come from the checkpoint's tokenizer
        # order shipped as char2id.json, else fall back to transformers
        c2i_path = os.path.join(model_dir, "char2id.json")
        if os.path.exists(c2i_path):
            self.char2id = json.load(open(c2i_path, encoding="utf-8"))
        else:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(
                os.path.join(os.path.dirname(model_dir), "tokenizer"))
            self.char2id = {t: i for t, i in tok.get_vocab().items()}
        self.tonal = {}
        for line in open(table_path, encoding="utf-8"):
            s, _, r = line.rstrip("\n").partition("\t")
            if r:
                self.tonal[s] = list(r)
        # toneless lookup: base syllable -> merged candidate chars
        self.toneless = {}
        for s, chars in self.tonal.items():
            b = "".join(c for c in s if c not in TONES)
            dst = self.toneless.setdefault(b, [])
            for ch in chars:
                if ch not in dst:
                    dst.append(ch)

    def cands(self, syl):
        """Legal (char, id) pairs for one bopomofo syllable."""
        has_tone = any(c in TONES for c in syl)
        chars = (self.tonal.get(syl) if has_tone
                 else self.toneless.get(syl)) or []
        return [(c, self.char2id[c]) for c in chars if c in self.char2id]

    def decode(self, syllables, n):
        sid = np.array([[self.syl_vocab.get(s, 1) for s in syllables]],
                       dtype=np.int64)
        amask = np.ones_like(sid, dtype=bool)
        lg = self.sess.run(None, {"syl": sid, "amask": amask})[0][0]
        best, margins = [], []          # per position: char + (margin, runner-up)
        for i, s in enumerate(syllables):
            cs = self.cands(s)
            if not cs:                  # unknown syllable: no legal decode
                return []
            scored = sorted(((lg[i, tid], ch) for ch, tid in cs), reverse=True)
            best.append(scored[0][1])
            if len(scored) > 1:
                margins.append((scored[0][0] - scored[1][0], i, scored[1][1]))
        out = ["".join(best)]
        for _, i, alt in sorted(margins)[: max(0, n - 1)]:
            s = best[:]
            s[i] = alt
            out.append("".join(s))
        return out

    def handle(self, req):
        syllables = req.get("syllables") or []
        n = int(req.get("n") or 1)
        if not syllables:
            return {"sentences": []}
        return {"sentences": self.decode(syllables, n)}


def read_request(conn):
    conn.settimeout(5.0)               # a stalled client must not wedge the loop
    buf = b""
    while b"\n" not in buf:
        chunk = conn.recv(4096)
        if not chunk:
            break
        buf += chunk
        if len(buf) > 1 << 20:
            return None
    return buf.split(b"\n", 1)[0] if buf else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model/slothe_4m_onnx")
    ap.add_argument("--table", default="model/phonetic_table.tsv")
    ap.add_argument("-s", "--socket", default=default_socket_path())
    args = ap.parse_args()

    dec = Decoder(args.model, args.table)
    # warm up so the first keystroke isn't slow
    dec.decode(["ㄋㄧˇ", "ㄏㄠˇ"], 1)

    if os.path.exists(args.socket):
        os.unlink(args.socket)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(args.socket)
    os.chmod(args.socket, 0o600)
    srv.listen(8)
    print(f"slothingd-e listening on {args.socket} "
          f"(model {args.model})", file=sys.stderr, flush=True)

    while True:
        conn, _ = srv.accept()
        try:
            raw = read_request(conn)
            if not raw:
                continue
            try:
                resp = dec.handle(json.loads(raw.decode("utf-8")))
            except Exception as e:      # bad request must not kill the daemon
                resp = {"sentences": [], "error": str(e)}
            conn.sendall((json.dumps(resp, ensure_ascii=False) + "\n").encode())
        except OSError:
            pass
        finally:
            conn.close()


if __name__ == "__main__":
    main()
