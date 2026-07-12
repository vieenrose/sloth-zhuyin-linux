import json, numpy as np, sys
TONES = set("ˊˇˋ˙")
strip = lambda s: "".join(c for c in s if c not in TONES)
# Candidate A: 10 phonetic group-keys (3x4 grid), covers all 37 bopomofo
A = ["ㄅㄆㄇㄈ","ㄉㄊㄋㄌ","ㄍㄎㄏ","ㄐㄑㄒ","ㄓㄔㄕㄖ","ㄗㄘㄙ",
     "ㄧㄨㄩ","ㄚㄛㄜㄝ","ㄞㄟㄠㄡ","ㄢㄣㄤㄥㄦ"]
sym2grp = {c: i for i, g in enumerate(A) for c in g}
def gclass(syl):
    return tuple(sym2grp[c] for c in strip(syl))  # tone dropped, per-symbol group

sv = json.load(open("syl_vocab.json"))            # str -> id
inv = {i: s for s, i in sv.items()}
N = len(sv)
specials = {s: i for s, i in sv.items() if s.startswith("<")}
maxspec = max(specials.values()) if specials else -1

# distinct group-classes -> new ids (after specials)
class_id, new_vocab = {}, dict(specials)
nxt = maxspec + 1
skipped = 0
for s, i in sv.items():
    if s.startswith("<"): continue
    try: k = gclass(s)
    except KeyError: skipped += 1; continue
    if k not in class_id:
        class_id[k] = nxt; new_vocab["G_" + "_".join(map(str, k))] = nxt; nxt += 1

remap = np.zeros(N, dtype=np.int64)
unk = specials.get("<unk>", 0)
for s, i in sv.items():
    if s.startswith("<"): remap[i] = i
    else:
        try: remap[i] = class_id[gclass(s)]
        except KeyError: remap[i] = unk

# transform the .bin (remap syllable IDs; leave char IDs untouched)
data = np.fromfile("train_e_g2pw.bin", dtype=np.uint16)
out = data.copy()
i = nrec = 0
while i < len(data):
    n = int(data[i]); s = i + 1
    out[s:s + n] = remap[data[s:s + n].astype(np.int64)].astype(np.uint16)
    i += 1 + 2 * n; nrec += 1
out.tofile("train_grouped.bin")
json.dump(new_vocab, open("grouped_syl_vocab.json", "w"), ensure_ascii=False)
# also save the syllable->class map + inverse legal-char lookup helper for the gate
json.dump({s: class_id.get(gclass(s), unk) for s in sv if not s.startswith("<")},
          open("syl2class.json", "w"), ensure_ascii=False)
print(f"records: {nrec}  orig syl-vocab: {N}  grouped vocab: {len(new_vocab)} "
      f"({len(class_id)} classes + {len(specials)} specials)  skipped: {skipped}")
# sanity: a few collapses
for ex in ["ㄅㄚ","ㄆㄚ","ㄉㄚ","ㄊㄚˋ","ㄓㄨㄤ","ㄔㄨㄤ"]:
    if ex in sv: print(f"  {ex} -> class {class_id.get(gclass(ex))} {gclass(ex)}")
