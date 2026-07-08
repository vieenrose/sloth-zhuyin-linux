# sloth-zhuyin-linux

An LLM-enhanced Zhuyin (Bopomofo) input method for fcitx5 on Ubuntu/Kubuntu.

- `engine/fcitx5-chewing/` — **Slothing**, an fcitx5 input method addon forked
  from [fcitx5-chewing](https://github.com/fcitx/fcitx5-chewing) (pinned to
  5.0.14) and re-namespaced so it installs and runs side-by-side with the
  stock `fcitx5-chewing`/`fcitx5-zhuyin` packages. Uses libchewing for
  zhuyin-key → candidate generation.
- `engine/slothingd/` — **slothingd**, a small Unix-socket daemon linking
  directly against llama.cpp's C API (`llama.h`, no `llama-server`, no
  Python bindings) that reranks libchewing's candidate list using
  grammar-constrained decoding, fixing the classic homophone-disambiguation
  weakness of n-gram based engines. It runs on a **pull** model: type your
  whole sentence as usual, then press the **convert key** (default **Ctrl+Enter**) to
  get the LLM's alternatives in a normal fcitx5 candidate list — number keys /
  arrows select, Enter/Space commits the highlighted one, Esc cancels. Nothing
  runs during ordinary typing, and the LLM never rewrites text on its own; you
  always pick from the list. Accepting a candidate also teaches libchewing the
  correction (via `chewing_userphrase_add`) so it converges over time. The
  daemon binds a per-user socket under `$XDG_RUNTIME_DIR`.
- `llm/` — local LLM runtime: a llama.cpp checkout (built for its headers
  and shared libs, which both `slothingd` and the engine link against) plus
  the GGUF model. Not vendored in git (see `.gitignore`); see setup below.
- `model/` — **SlothLM**, a ~34M-parameter zhuyin-native model trained from
  scratch for this IME (`model/DESIGN.md`): custom tokenizer (one token per
  bopomofo symbol / Han character), trained on Traditional-Chinese text plus
  synthetic conversion tasks in the exact serving format. **Much smaller than
  the interim LFM2.5-230M** (~7× fewer parameters) — the point is speed:
  conversion should feel instant (~50-150ms target vs ~200-350ms today),
  fast enough to also power per-phrase reranking and, eventually,
  direct zhuyin→sentence decoding. Includes `phonetic_table.tsv`
  (bopomofo syllable → legal Traditional characters), the legality
  constraint for the libchewing-free decode mode.
- `RESEARCH.md` — notes on how LLMs are used to improve input methods
  elsewhere, gathered before designing the reranker.
- `ARCHITECTURE.md` — how the engine, daemon, model, and eval fit together.
- `MODEL_BENCHMARKS.md` — comparison of candidate reranker models; see
  `MODEL_BENCHMARKS.md` for why **LFM2.5-230M (Q4_0 quant)** was picked.

## Status

Milestone 1 (plain fcitx5 zhuyin engine, no LLM) is done. Milestone 2 (LLM
conversion) was hardened after an adversarial code review found a cluster of
security/lifetime/UX bugs in the first async-suggestion design; it is now a
**pull-model conversion** with a hardened daemon (per-user socket,
SIGPIPE-safe, read-timeout) and a lifetime-safe background worker. The
conversion UX is **segment-based** (Japanese-henkan style): type
ㄨㄛˇㄗㄞˋㄔㄨㄥˊㄒㄧㄣㄎㄠˇㄌㄩˋ, press Ctrl+Enter → the LLM's best sentence
(我在重新考慮 where raw chewing produces 我再重新考慮) appears with the
focused phrase highlighted; ←/→ move between phrases, ↑/↓ cycle a phrase's
candidates with the sentence updating live, Enter commits, Esc cancels.

The current model is an interim off-the-shelf LFM2.5-230M; **SlothLM, a much
smaller (~34M) purpose-trained replacement, is in training** — see the
roadmap. The daemon also already has a libchewing-free *decode* mode
(`-t model/phonetic_table.tsv`, `{"syllables": [...]}` requests) that decodes
bopomofo directly under a phonetic-legality grammar, including toneless
input; it awaits SlothLM to be useful (`eval/run_decode_eval.py` is the
scoreboard).

## Roadmap

Informed by a verified research pass over the LLM-IME literature and the
zhuyin open-source ecosystem (`RESEARCH-LLM-IME.md`). As far as we could
verify, no other open-source zhuyin IME has an LLM component — McBopomofo,
vChewing, and libchewing are all purely statistical — so the goal is to keep
the "local, private, provably-safe" positioning while closing the accuracy
gap to commercial cloud systems (iFlytek's GeneInput reports P@1 88.4 vs ~71
for traditional conversion, with a 2.6B cloud model; we target the on-device
niche they left open). Notably, even Apple Intelligence's Traditional
Chinese debut (iOS 26.1, Nov 2025) only layers post-typing assistance
(Writing Tools, translation) *around* the keyboard — the zhuyin
conversion/選字 step itself remains unimproved by every major vendor, and
the one commercial local-first AI zhuyin keyboard we found (Keyly) is
closed-source and iOS-only. That conversion step, on desktop Linux, in the
open, is exactly Slothing's lane.

**Done**
- [x] fcitx5 engine (fork of fcitx5-chewing, side-by-side installable)
- [x] `slothingd`: local llama.cpp daemon, grammar-constrained decoding
      (output provably built from real libchewing candidates)
- [x] Pull-model conversion UX: convert key → native candidate list
- [x] Model selection by benchmark: LFM2.5-230M Q4_0 (95% char accuracy,
      ~200-350ms/request; see `MODEL_BENCHMARKS.md`)
- [x] Surrounding-text context: the LLM sees the document before the cursor
- [x] Tail-window conversion for long sentences (front becomes context)
- [x] Learning on accept: tone-correct `chewing_userphrase_add` so chewing
      itself converges
- [x] Hardened daemon (per-user `$XDG_RUNTIME_DIR` socket, SIGPIPE-safe,
      read timeouts) and lifetime-safe worker threading

**Next (v0.3) — accuracy *and* speed: SlothLM**
- [~] Replace the interim LFM2.5-230M with **SlothLM**, a ~34M from-scratch
      zhuyin-native model (`model/DESIGN.md`) — **~7× smaller**, targeting
      ~50-150ms per conversion instead of ~200-350ms. Task-specialising is
      **necessary, not optional** for accuracy too: on the 159-case eval the
      *un-fine-tuned* LFM2.5-230M top-1 (53% sentence) is a net regression vs
      chewing (61%); it wins only on recall (right answer in the list: 73% vs
      61%). A small model trained *on the serving format* converts that
      recall headroom into top-1 while being fast enough for features a 230M
      model can't serve (per-phrase reranking, typing-time hints, direct
      decoding). Off-the-shelf LLMs align phonetics poorly until trained on
      conversion (Huawei PY-GEC: cosine 0.26 → 0.82). Ship gates: top-1 ≥
      chewing's 61%, recall ≥ 73%, median latency < 500ms. Fallback track: a
      LoRA fine-tune of LFM2.5 (`finetune/`). Training in progress on GPU.
- [x] Evaluation harness: a scored zhuyin→sentence test set (per-char and
      per-sentence accuracy, latency) run against the daemon in CI, so model
      and prompt changes are measured instead of eyeballed.

**Then (v0.4) — personalization, kept local**
- [ ] LLM-rank the per-phrase candidate window (Down key). Today Down opens
      chewing's frequency-ranked homophones for the phrase at the cursor; once
      SlothLM is fast enough (~50-150ms) the daemon can reorder just that
      phrase's candidates by sentence context (a smaller query than
      whole-sentence Ctrl+Enter), so the quick single-word fix becomes
      context-aware too. Gated on SlothLM latency.
- [ ] Log accepted vs. shown-but-rejected conversion candidates on-device
      (opt-in) and use them as preference signal — the GeneInput RLHF recipe,
      but private: nothing leaves the machine.
- [ ] Per-user phrase bias in reranking (beyond what chewing's own learning
      captures).

**Later (v0.5+) — looser input, toward a libchewing-free decoder**
- [~] Direct zhuyin→sentence decoding (no libchewing in the loop): the daemon
      infrastructure is **built** — `slothingd -t model/phonetic_table.tsv`
      accepts `{"syllables": [...]}` and decodes under a grammar of
      phonetically-legal characters, tonal or toneless. The probe
      (`eval/run_decode_eval.py`) scores it against the chewing baseline;
      gated on SlothLM (the untrained interim model scores ~0%, by design —
      SlothLM's z2t/toneless training tasks are exactly this format).
- [ ] Tone-free zhuyin mode in the IME: type without tone keys (~35% fewer
      keystrokes), let the model disambiguate — the decode mode's toneless
      path already implements the backend.
- [ ] Typo tolerance for adjacent-key zhuyin mistakes (widen each syllable's
      legal set to keyboard-adjacent readings; same grammar mechanism).
- [ ] Packaging: .deb + a one-command setup script (llama.cpp build + model
      download), so non-developers can install.
- [ ] (Exploratory) Writing-tools actions on the preedit — a second key that
      offers rewrite/formalize variants of the composed sentence, Keyly-style
      but local and open; only worth pursuing once conversion accuracy is
      where we want it.

**Non-goals**
- Cloud inference of any kind; telemetry. The daemon binds a per-user socket
  and everything runs locally, always.
- Ripping out libchewing *before the model earns it*: the hybrid
  (statistical hot path + LLM on demand) stays the shipped configuration —
  it is also the industry pattern (PERT + n-gram + lexicon) — until the
  decode probe shows SlothLM beating chewing outright. The libchewing-free
  path is explored in measured stages behind the eval harness, never by
  faith.

## Build & install the fcitx5 addon

```sh
cd engine/fcitx5-chewing
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr
make -j"$(nproc)"
sudo make install
```

Restart fcitx5 (`fcitx5 -r -d`) and add "Slothing" (懶) as an input method via
`fcitx5-configtool`.

## Set up the local LLM runtime and build slothingd

One command — clones and builds llama.cpp, downloads the model, and builds
`slothingd` (idempotent, safe to re-run):

```sh
sh scripts/setup-llm.sh
```

<details><summary>...or the equivalent manual steps</summary>

```sh
mkdir -p llm && cd llm
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
cmake -B llama.cpp/build -S llama.cpp -DCMAKE_BUILD_TYPE=Release
cmake --build llama.cpp/build -j"$(nproc)" --target llama

hf download LiquidAI/LFM2.5-230M-GGUF LFM2.5-230M-Q4_0.gguf \
  --local-dir models/lfm2.5-230m-q4
cd ..

cmake -B engine/slothingd/build -S engine/slothingd -DCMAKE_BUILD_TYPE=Release
cmake --build engine/slothingd/build -j"$(nproc)"
```
</details>

Run it with `packaging/run-slothingd.sh` (manual start; no auto-start/systemd
unit by design — run it yourself when you want the reranker active).

## Development

`.github/workflows/ci.yml` builds the addon + daemon and runs the evaluation
harness (`eval/`) on every push, gating on LLM top-1 character accuracy so a
model or prompt change that breaks the pipeline fails CI.
