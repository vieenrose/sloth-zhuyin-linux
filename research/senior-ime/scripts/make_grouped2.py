import json, numpy as np
TONES = set("ˊˇˋ˙")
strip = lambda s: "".join(c for c in s if c not in TONES)
tone_of = lambda s: "".join(c for c in s if c in TONES) or "1"   # tone-1 = "1"
G10 = ["ㄅㄆㄇㄈ","ㄉㄊㄋㄌ","ㄍㄎㄏ","ㄐㄑㄒ","ㄓㄔㄕㄖ","ㄗㄘㄙ","ㄧㄨㄩ","ㄚㄛㄜㄝ","ㄞㄟㄠㄡ","ㄢㄣㄤㄥ","ㄦ"]
G21 = ["ㄅㄆ","ㄇㄈ","ㄉㄊ","ㄋㄌ","ㄍㄎ","ㄏ","ㄐㄑ","ㄒ","ㄓㄔ","ㄕㄖ","ㄗㄘ","ㄙ","ㄧㄨ","ㄩ","ㄚㄛ","ㄜㄝ","ㄞㄟ","ㄠㄡ","ㄢㄣ","ㄤㄥ","ㄦ"]

sv = json.load(open("syl_vocab.json"))
data = np.fromfile("train_e_g2pw.bin", dtype=np.uint16)

def build(name, groups, tone_kept):
    m = {c: i for i, g in enumerate(groups) for c in g}
    def sig(syl):
        try: g = tuple(m[c] for c in strip(syl))
        except KeyError: return None
        return "S_" + "_".join(map(str, g)) + ("|" + tone_of(syl) if tone_kept else "")
    specials = {s: i for s, i in sv.items() if s.startswith("<")}
    maxspec = max(specials.values()) if specials else -1
    vocab = dict(specials); class_id = {}; nxt = maxspec + 1
    for s in sv:
        if s.startswith("<"): continue
        k = sig(s)
        if k is None: continue
        if k not in class_id: class_id[k] = nxt; vocab[k] = nxt; nxt += 1
    remap = np.zeros(len(sv), dtype=np.int64)
    for s, i in sv.items():
        remap[i] = i if s.startswith("<") else class_id.get(sig(s), specials.get("<unk>", 0))
    out = data.copy(); i = 0
    while i < len(data):
        n = int(data[i]); st = i + 1
        out[st:st + n] = remap[data[st:st + n].astype(np.int64)].astype(np.uint16); i += 1 + 2 * n
    out.tofile(f"train_{name}.bin")
    json.dump(vocab, open(f"{name}_vocab.json", "w"), ensure_ascii=False)
    json.dump({"groups": groups, "tone_kept": tone_kept}, open(f"{name}_cfg.json", "w"), ensure_ascii=False)
    print(f"{name}: {len(class_id)} classes + {len(specials)} specials = {len(vocab)} input vocab (tone_kept={tone_kept})")

build("g10t", G10, True)
build("g21t", G21, True)
