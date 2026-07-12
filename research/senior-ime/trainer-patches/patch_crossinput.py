p = "train_slothe_ternary.py"
s = open(p, encoding="utf-8").read()
if "teacher_path" in s:
    print("already patched"); raise SystemExit

# 1) AlignedBin.__init__: accept a teacher data path (exact-syllable .bin)
s = s.replace(
    "    def __init__(self, path, ctx_p=0.0, typo=None, typo_p=0.0):\n"
    "        self.data = np.fromfile(path, dtype=np.uint16)\n"
    "        self.ctx_p, self.typo, self.typo_p = ctx_p, typo, typo_p",
    "    def __init__(self, path, ctx_p=0.0, typo=None, typo_p=0.0, teacher_path=None):\n"
    "        self.data = np.fromfile(path, dtype=np.uint16)\n"
    "        self.tdata = np.fromfile(teacher_path, dtype=np.uint16) if teacher_path else None\n"
    "        self.ctx_p, self.typo, self.typo_p = ctx_p, typo, typo_p")

# 2) __getitem__: return teacher-input syllables (exact) alongside student (grouped)
s = s.replace(
    "        s, n = self.idx[k]\n"
    "        syl = self.data[s:s + n].astype(np.int64)\n"
    "        chr_ = self.data[s + n:s + 2 * n].astype(np.int64)",
    "        s, n = self.idx[k]\n"
    "        syl = self.data[s:s + n].astype(np.int64)\n"
    "        tsyl = (self.tdata[s:s + n].astype(np.int64)\n"
    "                if self.tdata is not None else syl.copy())\n"
    "        chr_ = self.data[s + n:s + 2 * n].astype(np.int64)")
s = s.replace(
    "                syl = np.concatenate([np.zeros(L, dtype=np.int64), syl])\n"
    "                chr_ = np.concatenate([np.full(L, -100, dtype=np.int64), chr_])",
    "                syl = np.concatenate([np.zeros(L, dtype=np.int64), syl])\n"
    "                tsyl = np.concatenate([np.zeros(L, dtype=np.int64), tsyl])\n"
    "                chr_ = np.concatenate([np.full(L, -100, dtype=np.int64), chr_])")
s = s.replace(
    "        return torch.from_numpy(syl), torch.from_numpy(chr_), torch.from_numpy(forced)",
    "        return (torch.from_numpy(syl), torch.from_numpy(tsyl),\n"
    "                torch.from_numpy(chr_), torch.from_numpy(forced))")

# 3) collate: carry tsyl through
s = s.replace(
    "def collate(batch, pad=0):\n"
    "    n = max(len(s) for s, _, _ in batch); B = len(batch)\n"
    "    syl = torch.zeros(B, n, dtype=torch.long)\n"
    "    chr_ = torch.full((B, n), -100, dtype=torch.long)\n"
    "    mask = torch.zeros(B, n, dtype=torch.bool)\n"
    "    forced = torch.zeros(B, n, dtype=torch.long)\n"
    "    for i, (sq, c, f) in enumerate(batch):\n"
    "        syl[i, :len(sq)] = sq; chr_[i, :len(c)] = c; mask[i, :len(sq)] = True; forced[i, :len(f)] = f\n"
    "    return syl, chr_, mask, forced",
    "def collate(batch, pad=0):\n"
    "    n = max(len(s) for s, _, _, _ in batch); B = len(batch)\n"
    "    syl = torch.zeros(B, n, dtype=torch.long)\n"
    "    tsyl = torch.zeros(B, n, dtype=torch.long)\n"
    "    chr_ = torch.full((B, n), -100, dtype=torch.long)\n"
    "    mask = torch.zeros(B, n, dtype=torch.bool)\n"
    "    forced = torch.zeros(B, n, dtype=torch.long)\n"
    "    for i, (sq, tq, c, f) in enumerate(batch):\n"
    "        syl[i, :len(sq)] = sq; tsyl[i, :len(tq)] = tq\n"
    "        chr_[i, :len(c)] = c; mask[i, :len(sq)] = True; forced[i, :len(f)] = f\n"
    "    return syl, tsyl, chr_, mask, forced")

# 4) training loop: unpack tsyl, feed teacher the exact syllables
s = s.replace(
    "        for syl, chr_, mask, forced in dl:\n"
    "            syl, chr_, mask, forced = syl.to(dev), chr_.to(dev), mask.to(dev), forced.to(dev)",
    "        for syl, tsyl, chr_, mask, forced in dl:\n"
    "            syl, tsyl, chr_, mask, forced = (syl.to(dev), tsyl.to(dev),\n"
    "                chr_.to(dev), mask.to(dev), forced.to(dev))")
s = s.replace(
    "                        t_logits = teacher(syl, mask, hints)",
    "                        t_logits = teacher(tsyl, mask, hints)  # exact syllables")

# 5) --teacher-data arg + wire into AlignedBin
s = s.replace(
    '    ap.add_argument("--teacher", default="", help="dir with a fp teacher slothe.pt")',
    '    ap.add_argument("--teacher", default="", help="dir with a fp teacher slothe.pt")\n'
    '    ap.add_argument("--teacher-data", default="", help="exact-syllable .bin for cross-input distillation (teacher input)")')
s = s.replace(
    "    ds = AlignedBin(args.data, ctx_p=args.context, typo=typo_map, typo_p=args.typo_noise)",
    "    ds = AlignedBin(args.data, ctx_p=args.context, typo=typo_map, typo_p=args.typo_noise,\n"
    "                    teacher_path=(args.teacher_data or None))")

open(p, "w", encoding="utf-8").write(s)
print("patched: cross-input distillation (teacher=exact, student=grouped)")
