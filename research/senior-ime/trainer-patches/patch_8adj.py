p = "train_slothe_ternary.py"
s = open(p, encoding="utf-8").read()
if "_dachen_key_adjacency" in s:
    print("already patched"); raise SystemExit

helper = '''def _dachen_key_adjacency():
    """Bopomofo symbol -> set of symbols on physically 8-adjacent keys, per the
    TAAI-2024 keyboard 8-adjacency error model (Li/Yeh/Chang, 大城市 = 284t/6g4):
    a real mis-key lands on one of the 8 neighbours in the QWERTY 3x3 block
    ('h' -> t/y/u/g/j/b/n/m). Used to make typo-noise reflect finger slips
    instead of arbitrary edit-distance-1 substitutions."""
    rows = ["1234567890-=", "qwertyuiop[]", "asdfghjkl;'", "zxcvbnm,./"]
    pos = {k: (r, c) for r, row in enumerate(rows) for c, k in enumerate(row)}
    key_nb = {}
    for k, (r, c) in pos.items():
        key_nb[k] = {k2 for k2, (r2, c2) in pos.items()
                     if abs(r2 - r) <= 1 and abs(c2 - c) <= 1 and k2 != k}
    DACHEN = {"1": "\\u3105", "q": "\\u3106", "a": "\\u3107", "z": "\\u3108",
              "2": "\\u3109", "w": "\\u310a", "s": "\\u310b", "x": "\\u310c",
              "e": "\\u310d", "d": "\\u310e", "c": "\\u310f", "r": "\\u3110",
              "f": "\\u3111", "v": "\\u3112", "5": "\\u3113", "t": "\\u3114",
              "g": "\\u3115", "b": "\\u3116", "y": "\\u3117", "h": "\\u3118",
              "n": "\\u3119", "u": "\\u3127", "j": "\\u3128", "m": "\\u3129",
              "8": "\\u311a", "i": "\\u311b", "k": "\\u311c", ",": "\\u311d",
              "9": "\\u311e", "o": "\\u311f", "l": "\\u3120", ".": "\\u3121",
              "0": "\\u3122", "p": "\\u3123", ";": "\\u3124", "/": "\\u3125",
              "-": "\\u3126"}
    sym_key = {v: k for k, v in DACHEN.items()}
    return {sym: {DACHEN[k2] for k2 in key_nb.get(k, ()) if k2 in DACHEN}
            for sym, k in sym_key.items()}


def build_typo_neighbors(syl_vocab):'''
s = s.replace("def build_typo_neighbors(syl_vocab):", helper, 1)

# inject sym_adj + constrain the substitution case to 8-adjacency
s = s.replace(
    '    TONES = set("ˊˇˋ˙"); by_key = {}',
    '    TONES = set("ˊˇˋ˙"); by_key = {}\n    sym_adj = _dachen_key_adjacency()  # TAAI 8-adjacency slip model',
    1)
old_sub = '''                if len(a) == len(b):
                    if sum(x != y for x, y in zip(a, b)) == 1: nb.append(j)'''
new_sub = '''                if len(a) == len(b):
                    diffs = [(x, y) for x, y in zip(a, b) if x != y]
                    if len(diffs) == 1 and diffs[0][1] in sym_adj.get(diffs[0][0], ()):
                        nb.append(j)  # only physically-adjacent-key slips'''
assert old_sub in s, "substitution anchor not found"
s = s.replace(old_sub, new_sub, 1)
open(p, "w", encoding="utf-8").write(s)
print("patched: 8-adjacency-constrained typo substitution")
