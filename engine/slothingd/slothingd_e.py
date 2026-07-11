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


LEARN_PATH = os.path.expanduser("~/.local/share/slothing/learn.tsv")


class Decoder:
    def __init__(self, model_dir, table_path):
        # user adaptation: syllable->picked char, syllable-pair->picked phrase.
        # Applied as LOGIT BONUSES calibrated on measured gaps (flips ties+
        # moderate homophones ~5, spares strong context ~10+: 的, 重新);
        # evidence still wins), not hard overrides.
        self.learn_char = {}
        self.learn_phrase = {}
        try:
            for line in open(LEARN_PATH, encoding="utf-8"):
                kind, key, val = line.rstrip("\n").split("\t")
                if kind == "c":
                    self.learn_char[key] = val
                elif kind == "p":
                    self.learn_phrase[tuple(key.split(" "))] = val
        except FileNotFoundError:
            pass
        onnx = os.path.join(model_dir, "model_quantized.onnx")
        if not os.path.exists(onnx):
            onnx = os.path.join(model_dir, "model.onnx")
        if not os.path.exists(onnx):
            sys.exit(f"slothingd-e: no model at {model_dir}/model_quantized.onnx"
                     " — run packaging/fetch-model.sh first")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2      # keystroke latency, not throughput
        self.sess = ort.InferenceSession(onnx, opts,
                                         providers=["CPUExecutionProvider"])
        # hint-conditioned models take a third input (user-corrected chars)
        self.has_hints = any(i.name == "hints" for i in self.sess.get_inputs())
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

    @staticmethod
    def _dl1(a, b):
        """Damerau-Levenshtein distance capped at 1 (typo tolerance)."""
        if a == b:
            return 0
        la, lb = len(a), len(b)
        if abs(la - lb) > 1:
            return 2
        if la == lb:
            diff = [i for i in range(la) if a[i] != b[i]]
            if len(diff) == 1:
                return 1
            if (len(diff) == 2 and diff[1] == diff[0] + 1
                    and a[diff[0]] == b[diff[1]] and a[diff[1]] == b[diff[0]]):
                return 1                             # transposition
            return 2
        s, l = (a, b) if la < lb else (b, a)         # one insert/delete
        return 1 if any(s[:i] + l[i] + s[i:] == l
                        for i in range(len(s) + 1)) else 2

    def cands(self, syl):
        """Legal (char, id) pairs for one bopomofo syllable. Unknown bases
        (key-slip typos producing impossible syllables) fall back to the
        union of all bases within edit distance 1 — the model's context
        picks. Legal syllables are unaffected."""
        has_tone = any(c in TONES for c in syl)
        chars = (self.tonal.get(syl) if has_tone
                 else self.toneless.get(syl)) or []
        return [(c, self.char2id[c]) for c in chars if c in self.char2id]

    def typo_fixes(self, syl):
        """For an impossible syllable, candidate corrections within edit
        distance 1: [(vocab_syllable, chars)] (tone of the typed syllable is
        preserved when the corrected base has it)."""
        tone = "".join(c for c in syl if c in TONES)
        base = "".join(c for c in syl if c not in TONES)
        out = []
        for b in self.toneless:
            if self._dl1(b, base) <= 1:
                if tone and (b + tone) in self.tonal:
                    out.append((b + tone, self.tonal[b + tone]))
                elif not tone and b in self.tonal:   # tone-1 is unmarked
                    out.append((b, self.tonal[b]))
                elif not tone:
                    out.append((b, self.toneless[b]))
        return out

    def save_learn(self):
        os.makedirs(os.path.dirname(LEARN_PATH), exist_ok=True)
        with open(LEARN_PATH, "w", encoding="utf-8") as f:
            for k, v in self.learn_char.items():
                f.write(f"c\t{k}\t{v}\n")
            for k, v in self.learn_phrase.items():
                f.write(f"p\t{' '.join(k)}\t{v}\n")

    def learn(self, req):
        for syl, ch in req.get("chars") or []:
            self.learn_char[syl] = ch
        for pair, ph in req.get("phrases") or []:
            self.learn_phrase[tuple(pair.split(" "))] = ph
        self.save_learn()
        return {"ok": True}

    def _bonus(self, syllables):
        """(position, char) -> additive logit bonus from learned picks."""
        b = {}
        # Calibrated 2026-07-11 on the 230-case 免選字 set WITH a heavily-used
        # learn store: 6/8 over-personalized (59% vs the raw model's 74% —
        # bonuses flipped correct strong-context picks); 2/3 recovers 74%
        # while still flipping genuine near-ties (personalized ㄧㄣ->音 needs
        # >1.0). Sweep: 6/8=59%, 3/4=71%, 2/3=74%+flip, 1/1.5=74% flip lost.
        CHAR_BONUS, PHRASE_BONUS = 2.0, 3.0
        for i, s in enumerate(syllables):
            ch = self.learn_char.get(s)
            if ch:
                b[(i, ch)] = b.get((i, ch), 0.0) + CHAR_BONUS
        for i in range(len(syllables) - 1):
            ph = self.learn_phrase.get((syllables[i], syllables[i + 1]))
            if ph and len(ph) >= 2:
                b[(i, ph[0])] = b.get((i, ph[0]), 0.0) + PHRASE_BONUS
                b[(i + 1, ph[1])] = b.get((i + 1, ph[1]), 0.0) + PHRASE_BONUS
        return b

    def _run(self, sid, amask, hints_vec=None):
        feed = {"syl": sid, "amask": amask}
        if self.has_hints:
            if hints_vec is None:
                hints_vec = np.zeros_like(sid)
            feed["hints"] = hints_vec
        return self.sess.run(None, feed)

    CTX_MAX = 12

    def decode(self, syllables, n, hints=None, context=""):
        """hints: {position: char} user corrections; context: committed text
        before the cursor. Both condition the decode (hint channel); with a
        context/hint-trained model the whole sentence re-scores around them."""
        # context chars ride the hint channel on <pad>-id prefix positions
        ctx = [c for c in context if c in self.char2id][-self.CTX_MAX:] \
            if (context and self.has_hints) else []
        L = len(ctx)
        hint_ids = None
        if (hints or ctx) and self.has_hints:
            hint_ids = np.zeros((1, L + len(syllables)), dtype=np.int64)
            for j, c in enumerate(ctx):
                hint_ids[0, j] = self.char2id[c] + 1
            for k, ch in (hints or {}).items():
                i = int(k)
                if 0 <= i < len(syllables) and ch in self.char2id:
                    hint_ids[0, L + i] = self.char2id[ch] + 1
        ids = [0] * L + [self.syl_vocab.get(s, 1) for s in syllables]
        cand_override = {}
        # typo correction: an impossible syllable (no legal chars) is replaced
        # by the edit-distance-1 correction the model itself scores highest —
        # each candidate is tried as real input in one batched forward.
        for i, s in enumerate(syllables):
            if self.cands(s):
                continue
            fixes = [(v, cs) for v, cs in self.typo_fixes(s)
                     if v in self.syl_vocab
                     and any(c in self.char2id for c in cs)]
            if not fixes:
                continue
            batch = np.tile(np.array(ids, dtype=np.int64), (len(fixes), 1))
            for j, (v, _) in enumerate(fixes):
                batch[j, L + i] = self.syl_vocab[v]
            am = np.ones_like(batch, dtype=bool)
            hb = np.tile(hint_ids, (len(fixes), 1)) if hint_ids is not None else None
            lgb = self._run(batch, am, hb)[0]
            best_j, best_v = 0, -np.inf
            typed_len = len([c for c in s if c not in TONES])
            for j, (fv, cs) in enumerate(fixes):
                v = max(lgb[j, L + i, self.char2id[c]] for c in cs
                        if c in self.char2id)
                # prior: a MISSED key (longer corrected base) is the common
                # slip; deletions (shorter base) drop typed information
                base_len = len([c for c in fv if c not in TONES])
                v += 1.5 * (base_len - typed_len)
                if v > best_v:
                    best_v, best_j = v, j
            ids[L + i] = self.syl_vocab[fixes[best_j][0]]
            cand_override[i] = [(c, self.char2id[c]) for c in fixes[best_j][1]
                                if c in self.char2id]
        sid = np.array([ids], dtype=np.int64)
        amask = np.ones_like(sid, dtype=bool)
        lg = self._run(sid, amask, hint_ids)[0][0][L:]   # drop ctx positions
        bonus = self._bonus(syllables)
        best, margins = [], []          # per position: char + (margin, runner-up)
        ranked = []                     # per position: chars by model score
        for i, s in enumerate(syllables):
            cs = cand_override.get(i) or self.cands(s)
            if not cs:                  # unknown syllable: no legal decode
                return [], []
            scored = sorted(((lg[i, tid] + bonus.get((i, ch), 0.0), ch)
                             for ch, tid in cs), reverse=True)
            ranked.append([ch for _, ch in scored])
            best.append(scored[0][1])
            if len(scored) > 1:
                margins.append((scored[0][0] - scored[1][0], i, scored[1][1]))
        out = ["".join(best)]
        for _, i, alt in sorted(margins)[: max(0, n - 1)]:
            s = best[:]
            s[i] = alt
            out.append("".join(s))
        return out, ranked

    def phrases(self, syllables, i, n):
        """Model-ranked 2-char phrases for positions i,i+1: per-position
        softmax over legal chars, cross-ranked by joint probability (same
        scheme as the web demo's buildPhrases)."""
        if not (0 <= i < len(syllables) - 1):
            return [], []
        c0, c1 = self.cands(syllables[i]), self.cands(syllables[i + 1])
        if not c0 or not c1:
            return [], []
        sid = np.array([[self.syl_vocab.get(s, 1) for s in syllables]],
                       dtype=np.int64)
        amask = np.ones_like(sid, dtype=bool)
        lg = self._run(sid, amask)[0][0]

        def top(cands, pos, k=5):
            v = np.array([lg[pos, tid] for _, tid in cands])
            e = np.exp(v - v.max()); p = e / e.sum()
            idx = np.argsort(-p)[:k]
            return [(cands[j][0], float(p[j])) for j in idx]

        scored = [(a + b, pa * pb)
                  for a, pa in top(c0, i) for b, pb in top(c1, i + 1)]
        scored.sort(key=lambda x: -x[1])
        cut = max(0.06, 0.15 * scored[0][1]) if scored else 0
        out, probs = [], []
        for ph, p in scored:
            if p < cut or len(out) >= n:
                break
            if ph not in out:
                out.append(ph)
                probs.append(p)
        return out, probs

    def handle(self, req):
        syllables = req.get("syllables") or []
        n = int(req.get("n") or 1)
        if not syllables:
            return {"sentences": []}
        if req.get("phrase_at") is not None:   # 2-char phrase candidates
            phrases, probs = self.phrases(syllables, int(req["phrase_at"]), n)
            return {"sentences": phrases, "scores": probs}
        sentences, ranked = self.decode(syllables, n, hints=req.get("hints"),
                                        context=req.get("context") or "")
        return {"sentences": sentences, "candidates": ranked}

    def handle_any(self, req):
        if req.get("learn") is not None:       # user picks -> persist
            return self.learn(req["learn"])
        return self.handle(req)


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
                resp = dec.handle_any(json.loads(raw.decode("utf-8")))
            except Exception as e:      # bad request must not kill the daemon
                resp = {"sentences": [], "error": str(e)}
            conn.sendall((json.dumps(resp, ensure_ascii=False) + "\n").encode())
        except OSError:
            pass
        finally:
            conn.close()


if __name__ == "__main__":
    main()
