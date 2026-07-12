p = "train_slothe_ternary.py"
s = open(p, encoding="utf-8").read()
if "--save-every" in s:
    print("already patched"); raise SystemExit
# 1) add the arg
arg_anchor = '    ap.add_argument("--context", type=float, default=0.0)'
assert arg_anchor in s
s = s.replace(arg_anchor, arg_anchor +
    '\n    ap.add_argument("--save-every", type=int, default=0,'
    ' help="snapshot {out}_ep{N} every N epochs (for accuracy-vs-epoch curve)")')
# 2) snapshot at epoch end (before the epoch-level break)
snap_anchor = '''        if step >= total:
            break

    raw_model.set_quant_alpha(1.0)'''
assert snap_anchor in s
snap = '''        if args.save_every and (ep + 1) % args.save_every == 0:
            if is_main:
                _snap = f"{args.out}_ep{ep + 1}"
                os.makedirs(_snap, exist_ok=True)
                torch.save({"model": raw_model.state_dict(),
                            "config": {"n_syl": len(syl_vocab), "n_char": len(tok),
                                       "dim": args.dim, "depth": args.depth,
                                       "heads": args.heads, "kv": args.kv_heads,
                                       "ffn": args.ffn, "embed_norm": args.embed_norm,
                                       "weight_bits": args.quant,
                                       "weight_quant": args.weight_quant,
                                       "fp_boundary": args.fp_boundary,
                                       "pre_norm": args.pre_norm,
                                       "act_bits": args.act_bits,
                                       "char_hints": args.char_hints,
                                       "tie_hints": args.tie_hints}},
                           os.path.join(_snap, "slothe.pt"))
                json.dump(syl_vocab, open(os.path.join(_snap, "syl_vocab.json"),
                          "w", encoding="utf-8"), ensure_ascii=False)
                print(f"snapshot saved to {_snap}", flush=True)
            if ddp: dist.barrier()
        if step >= total:
            break

    raw_model.set_quant_alpha(1.0)'''
s = s.replace(snap_anchor, snap)
open(p, "w", encoding="utf-8").write(s)
print("patched: --save-every + per-epoch snapshot")
