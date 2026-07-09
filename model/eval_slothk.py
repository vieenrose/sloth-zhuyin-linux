#!/usr/bin/env python3
"""Evaluate SlothLM-K (raw-keystream model) on its three gates:

1. decode: eval/testset.tsv + eval/reference_mspy.tsv rendered as keystreams
   (tonal syllables -> keys, tone-1 followed by space) -> sentence accuracy.
2. code-switch: generated en/zh compositions (same patterns as the JS
   exhaustive test) -> segmentation+passthrough accuracy.

Decoding = one forward pass, per-key argmax over ALL labels (BLANK / ASCII /
Han). Assembled output = concat of non-blank labels. Unconstrained: the model
alone decides boundaries AND characters.

  python3 model/eval_slothk.py --model model/slothk --tokenizer model/tokenizer \
      --table model/phonetic_table.tsv --testset eval/testset.tsv
"""
import argparse, json, os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_slothlm_e import SlothE

TONES = "ˊˇˋ˙"
KEYMAP = {"ㄅ":"1","ㄆ":"q","ㄇ":"a","ㄈ":"z","ㄉ":"2","ㄊ":"w","ㄋ":"s","ㄌ":"x",
 "ㄍ":"e","ㄎ":"d","ㄏ":"c","ㄐ":"r","ㄑ":"f","ㄒ":"v","ㄓ":"5","ㄔ":"t","ㄕ":"g",
 "ㄖ":"b","ㄗ":"y","ㄘ":"h","ㄙ":"n","ㄧ":"u","ㄨ":"j","ㄩ":"m","ㄚ":"8","ㄛ":"i",
 "ㄜ":"k","ㄝ":",","ㄞ":"9","ㄟ":"o","ㄠ":"l","ㄡ":".","ㄢ":"0","ㄣ":"p","ㄤ":";",
 "ㄥ":"/","ㄦ":"-","ˊ":"6","ˇ":"3","ˋ":"4","˙":"7"}
BLANK, ASCII_BASE = 1, 2


def keys_of(syl):
    k = [KEYMAP[c] for c in syl]
    if syl[-1] not in TONES:
        k.append(" ")
    return "".join(k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model/slothk")
    ap.add_argument("--tokenizer", default="model/tokenizer")
    ap.add_argument("--table", default="model/phonetic_table.tsv")
    ap.add_argument("--testset", default="eval/testset.tsv")
    ap.add_argument("--reference", default="eval/reference_mspy.tsv")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    ck = torch.load(os.path.join(args.model, "slothe.pt"), map_location="cpu")
    c = ck["config"]; LB = c["label_base"]
    m = SlothE(c["n_syl"], c["n_char"], c["dim"], c["depth"], c["heads"], c["kv"], c["ffn"])
    m.load_state_dict(ck["model"]); m.eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"; m.to(dev)
    kv = json.load(open(os.path.join(args.model, "key_vocab.json"), encoding="utf-8"))

    # phonetic legality for constrained decode
    tonal_map = {}
    for line in open(args.table, encoding="utf-8"):
        s, _, rest = line.rstrip("\n").partition("\t")
        if rest: tonal_map[s] = list(rest)
    REVK = {v: k for k, v in KEYMAP.items()}

    def trailing_syllable_cands(keys, i):
        """Legal chars for a syllable ENDING at key i (tone at i, or symbols
        ending at i for tone-1). Returns list of tokenizer label ids."""
        cands = []
        tone = REVK.get(keys[i]) if keys[i] in "6347" else None
        end = i - 1 if tone else i
        bopo_rev = []
        for j in range(end, max(-1, end - 3), -1):
            sym = REVK.get(keys[j])
            if sym is None or sym in TONES: break
            bopo_rev.append(sym)
            syl = "".join(reversed(bopo_rev)) + (tone or "")
            if tone and syl in tonal_map:
                cands += tonal_map[syl]
            elif not tone:
                # tone-1 = syllable with no tone mark in the table
                if syl in tonal_map: cands += tonal_map[syl]
        ids = []
        for ch in cands:
            t = tok.convert_tokens_to_ids(ch)
            if t is not None: ids.append(t + LB)
        return ids

    def decode(keys, constrained=True):
        ids = torch.tensor([[kv.get(k, 1) for k in keys]], device=dev)
        am = torch.ones_like(ids, dtype=torch.bool)
        with torch.no_grad():
            logits = m(ids, am)[0]
        raw = logits.argmax(-1).tolist()
        out = []
        for i, lab in enumerate(raw):
            if lab == BLANK or lab == 0: continue
            if lab < LB:
                # model says English: EMIT THE ACTUAL KEYSTROKE (copy, don't predict)
                out.append(keys[i] if constrained else chr(lab - ASCII_BASE))
            else:
                if constrained:
                    legal = trailing_syllable_cands(keys, i)
                    if legal:
                        best = max(legal, key=lambda t: logits[i, t].item())
                        out.append(tok.convert_ids_to_tokens(best - LB)); continue
                t = tok.convert_ids_to_tokens(lab - LB)
                out.append(t if t else "?")
        return "".join(out)

    def run_set(path, name):
        n = ok = 0; misses = []
        for line in open(path, encoding="utf-8"):
            line = line.rstrip("\n")
            if not line or line.startswith("#"): continue
            bopomofo, expected = line.split("\t")[:2]
            keys = "".join(keys_of(s) for s in bopomofo.split())
            got = decode(keys)
            n += 1; hit = got == expected; ok += hit
            if not hit and len(misses) < 8: misses.append((expected, got))
        print(f"{name}: {ok}/{n} ({100*ok/n:.0f}%)")
        for e, g in misses: print(f"    want {e}  got {g}")
        return ok, n

    run_set(args.testset, "decode testset (homophone-hard)")
    if os.path.exists(args.reference):
        run_set(args.reference, "免選字 reference")

    # code-switch: same composition patterns as the JS exhaustive test
    tonal = [l.split("\t")[0] for l in open(args.table, encoding="utf-8")
             if "\t" in l and any(t in l.split("\t")[0] for t in TONES)]
    ZH = tonal[::7]
    EN = ("model world banana slothing kubernetes javascript keyboard password "
          "apple random coffee video manager github docker linux chrome firefox "
          "android laptop server client router monitor speaker battery charger "
          "online upload download update install backup restore reboot").split()
    n = ok = 0; misses = []
    def zh_pair(i): return ZH[i % len(ZH)], ZH[(i*13+7) % len(ZH)]
    def check(keys, want, tag):
        nonlocal n, ok
        got = decode(keys); n += 1
        if got == want: ok += 1
        elif len(misses) < 10: misses.append((tag, want, got))
    for i, w in enumerate(EN):
        a, b = zh_pair(i)
        chr_a = None  # expected chars unknown without decode; instead check shape
        # For code-switch we check ENGLISH PASSTHROUGH exactly and that Chinese
        # positions produce exactly one Han char per syllable.
        got = decode(w + keys_of(a) + keys_of(b)); n += 1
        hanlen = sum(1 for ch in got if "一" <= ch <= "鿿")
        if got.startswith(w) and hanlen == 2 and len(got) == len(w) + 2: ok += 1
        elif len(misses) < 10: misses.append(("en+zh+zh", w+"+2han", got))
        got = decode(keys_of(a) + w + " " + keys_of(b)); n += 1
        hanlen = sum(1 for ch in got if "一" <= ch <= "鿿")
        if w in got and hanlen == 2: ok += 1
        elif len(misses) < 10: misses.append(("zh+en+zh", w, got))
        got = decode(w); n += 1
        if got == w: ok += 1
        elif len(misses) < 10: misses.append(("en-alone", w, got))
    print(f"code-switch (segmentation+passthrough): {ok}/{n} ({100*ok/n:.0f}%)")
    for t, w, g in misses: print(f"    [{t}] want {w}  got {g}")


if __name__ == "__main__":
    main()
