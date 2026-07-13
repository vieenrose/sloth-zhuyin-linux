# Architecture

Sloth IME(樹懶智慧輸入法,全名「樹懶注音輸入法」)is an LLM-assisted Zhuyin (Bopomofo) input method for fcitx5 on
Linux. It keeps libchewing as the fast, proven decoder and adds an on-demand
local-LLM conversion step that reranks libchewing's own candidates. Nothing
leaves the machine.

## The pieces

```
   keystrokes
       │
       ▼
┌──────────────────────┐        Ctrl+Enter (convert key)
│ fcitx5 engine        │  ───────────────────────────────┐
│ engine/fcitx5-chewing│                                  │
│  (libsloth.so)    │                                  ▼
│  - libchewing decode │                        ┌───────────────────────┐
│  - ConvertState SM   │   harvest candidates   │ slothd daemon      │
│  - CommonCandidateList│  ────────────────────▶ │ engine/slothd      │
│    of LLM sentences  │   {positions, n, ctx}  │  - llama.cpp C API    │
│  - learns on accept  │ ◀──────────────────────│  - GBNF grammar over  │
└──────────────────────┘   {sentences:[...]}    │    the real candidates│
       │                    (Unix socket,        │  - LFM2.5-230M today; │
       ▼                     $XDG_RUNTIME_DIR)   │    SlothLM (model/)    │
   committed text                                └───────────────────────┘
```

### `engine/fcitx5-chewing/` — the fcitx5 addon (`libsloth.so`)
A fork of upstream fcitx5-chewing, re-namespaced so it installs side-by-side
with stock chewing. In **Composing** state it *is* stock chewing — no LLM work
during ordinary typing. The convert key (default Ctrl+Enter) moves through a
`ConvertState` machine (Composing → Converting → Choosing): it harvests
libchewing's per-interval candidates, sends one request to the daemon on a
joinable worker thread, and shows the returned sentences in a native
`CommonCandidateList`. Accepting one commits it and teaches libchewing the
correction (`chewing_userphrase_add`, tone-correct via `chewing_get_phoneSeq`).
Long sentences convert only a tail window; the prefix becomes context. The UI
(diff-highlighting, error feedback) is under separate iteration.

### `engine/slothd/` — the reranker daemon
A small C++ daemon linking llama.cpp's C API directly (no llama-server, no
Python). One request = one connection on a per-user `$XDG_RUNTIME_DIR` socket.
For each request it builds a **GBNF grammar** that admits exactly the
per-position candidates libchewing produced, so the model can only ever output
a real candidate combination — never hallucinated text. Returns N deduped
sentences (greedy + temperature passes). Hardened: SIGPIPE-ignored,
read-timeouts, bounded generation. The prompt/protocol format is **frozen**
because the SlothLM model is trained against it.

### `llm/` — the local runtime (not vendored)
A llama.cpp checkout (built for headers + libs the daemon links) plus the GGUF
model. Set up with `scripts/setup-llm.sh`. Runs via `packaging/run-slothd.sh`
(manual start by design). The daemon being down just means conversion is a
no-op; typing still works as stock chewing.

### `eval/` — the measurement + regression gate
`harvest.c` replicates the engine's exact candidate harvest against
libchewing; `run_eval.py` drives the 159-case `testset.tsv` through the live
daemon and reports chewing-vs-LLM top-1/recall/latency plus a confusion-pair
breakdown. `make_testset.py` validates new cases. CI
(`.github/workflows/ci.yml`) runs this on every push and gates on recall.

### `model/` — SlothLM, a from-scratch task-specialised LM (in progress)
The base LFM2.5-230M reranks below chewing at top-1 (it wins only on recall),
because none of its capacity is aimed at phonetics. `model/` builds a ~34M
LlamaForCausalLM trained from scratch on Traditional-Chinese text + synthetic
zhuyin↔text/SELECT/tone-free task data, with a custom byte-level BPE tokenizer
(one token per bopomofo symbol / common Han char, so GBNF aligns to token
boundaries). See `model/DESIGN.md`. Pipeline: `pull_corpus.py` →
`build_tokenizer.py` → `prepare_data.py` → `train_slothlm.py` →
`register_tokenizer.py` + convert_hf_to_gguf → `upload_to_hf.py`.

### `finetune/` — the LoRA alternative
Fine-tunes a generic model (LFM2.5) on the same task data. Kept as a lower-risk
fallback to the from-scratch SlothLM.

## Data flow of one conversion

1. User types zhuyin; libchewing forms a sentence (Composing).
2. User presses the convert key. The engine harvests, per phrase interval, the
   candidates libchewing itself offers (span-length-matched), and reads the
   surrounding text before the cursor as context.
3. It sends `{positions, n, context}` to the daemon.
4. The daemon builds a GBNF grammar from `positions`, prompts the model, and
   grammar-constrained-decodes N candidate sentences.
5. The engine shows them (plus chewing's own) in a candidate list; the user
   picks one, which commits and teaches chewing.

Every candidate the user ever sees is provably built from real libchewing
candidates — the safety invariant that makes a tiny model usable.

## Design rules

- **Local & private only.** No cloud inference, no telemetry.
- **libchewing stays the hot path.** The LLM is on-demand, never per-keystroke.
- **Grammar-constrained, always.** The model reorders real candidates; it never
  emits free text into the user's document.
- **Measured, not eyeballed.** Model/prompt changes are gated by `eval/`.
