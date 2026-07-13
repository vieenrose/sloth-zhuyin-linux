# Slothing(懶音)— an LLM-powered Zhuyin IME

**Type bopomofo; a model converts the whole sentence.** **Slothing** (Chinese
name **懶音**, from 樹懶 "sloth" + 注音 zhuyin; full zh name 樹懶注音輸入法):
a **25M-parameter ternary (W1.58A8)**
language model trained from scratch that decodes zhuyin to Traditional Chinese
locally — libchewing-free, with every character guaranteed to be a legal
reading of what you typed. Four frontends — desktop (fcitx5, IBus), Android,
and the browser — share one core, one model, and one **ggml inference core
(libslothe / TQ2_0), replacing ONNX Runtime**.

**中文說明(預設): [README.md](README.md)** ·
**Try it now (no install): [huggingface.co/spaces/Luigi/slothing-web](https://huggingface.co/spaces/Luigi/slothing-web)** ·
**Model: [Luigi/slothe-t-25m-zhuyin](https://huggingface.co/Luigi/slothe-t-25m-zhuyin)**

<p align="center"><img src="docs/demo-web-v13.gif" width="470" alt="Web demo (in-browser decode): typing 晚上熬夜看 world cup，白天在 louisa key-by-key on the virtual keyboard — correct whole-sentence 免選字, auto zh/en, each keypress highlighted live"></p>
<p align="center"><img src="docs/android-boox-demo-v11.gif" width="380" alt="Native Android IME (BOOX e-ink, 25M ternary model): typing 晚上熬夜看world cup,白天在louisa key-by-key — whole-sentence 免選字, auto zh/en switch (world cup / louisa kept English), each keypress highlighted live"></p>

## Highlights

| | |
|---|---|
| **免選字 sentence conversion** | 微軟新注音-style live conversion; **76%** on a held-out whole-sentence benchmark (honest held-out — see [docs/EVAL.md](docs/EVAL.md)) |
| **Auto zh/en** | No mode key: type `我用python寫程式` straight through — a DP segmenter decides |
| **Tone-optional** | Skip tone keys (~35% fewer keystrokes); context disambiguates |
| **Typo repair** | Impossible syllables fixed by the model (edit distance 1) |
| **聯想 prediction** | Next-word suggestions after commit (dictionary + personal habits): tap-to-chain on mobile, ⇧1-9 on desktop |
| **Fully offline** | **18 MB ternary GGUF** running on ggml / libslothe (TQ2_0 core) — ARM/x86/WASM, ~2× faster than int8, no cloud, no telemetry |

> **Pick re-scoring / document context** (the char-hint channel) is **v2**: v1
> ships the highest-accuracy no-hints model first; a hinted ternary model is
> retraining and this feature returns once it validates.

Sourced comparison vs Gboard 注音 and the Boox built-in IME:
**[docs/COMPARISON.md](docs/COMPARISON.md)** (zh-TW); 4-frontend UI logic matrix:
**[docs/UI-MATRIX.md](docs/UI-MATRIX.md)**.

## Install

Desktop platforms need the decode daemon (one-time setup, shared by fcitx5 + IBus),
running the ternary model on ggml:

```sh
# build the local ggml runtime first (llm/llama.cpp — see README "local LLM runtime"), then:
cmake -S engine/slothingd -B engine/slothingd/build_slothe -DCMAKE_BUILD_TYPE=Release
cmake --build engine/slothingd/build_slothe --target slothingd_slothe
packaging/fetch-model.sh                   # 18 MB ternary GGUF
packaging/install-slothingd-service.sh     # auto-start at login
```

| Platform | Install |
|---|---|
| **fcitx5** (KDE, …) | `.deb` from Releases, or `cmake -B engine/fcitx5-chewing/build -S engine/fcitx5-chewing -DCMAKE_INSTALL_PREFIX=/usr && cmake --build engine/fcitx5-chewing/build -j$(nproc) && sudo make -C engine/fcitx5-chewing/build install` |
| **IBus** (GNOME, …) | `.deb` from Releases, or the one-shot `engine/ibus-slothing/install.sh`; see `engine/ibus-slothing/README.md` |
| **Android** | `.apk` from Releases (**no daemon** on the phone — decoding runs on-device; the model ships inside the APK, no ONNX runtime, so the APK is ~20 MB smaller than before), or `cd android && ./gradlew :app:assembleDebug` (needs SDK/NDK) |
| **Browser** | nothing to install: [HF Space](https://huggingface.co/spaces/Luigi/slothing-web) |

## How it works

Zhuyin→Chinese is *aligned sequence labeling* (N syllables → N characters, each
constrained to its homophone set), so Slothing uses a **bidirectional encoder**
(non-autoregressive, one forward pass) instead of a causal LM: a **25M ternary
model** (each weight {−1,0,+1}×abs-median scale, int8 activations, QAT/STE;
boundary layers kept fp), trained on g2pW context-aware readings. A
dependency-free DP segmenter parses the keystream (auto zh/en), and decoding is
masked per position to legal readings.

All four frontends are thin adapters over the shared core in `engine/common`,
and share one **`libslothe`** inference implementation — a single ggml forward
pass (reusing mainline ggml's TQ2_0 ternary kernels): a native daemon on desktop,
NDK arm64 on Android, Emscripten WASM in the browser. Behavior is held together
by offline contract tests (core_test), a headless IBus end-to-end test, the
`eval/ui-parity` suite, and a per-layer / per-character golden check against PyTorch.

- Model, GGUF + full reproduction pipeline (data → labels → training → conversion):
  [Luigi/slothe-t-25m-zhuyin](https://huggingface.co/Luigi/slothe-t-25m-zhuyin)
- Architecture & design: `ARCHITECTURE.md`, `model/DESIGN-E.md`, `MODEL_BENCHMARKS.md`

## The model — why ternary

<p align="center"><img src="docs/score_vs_latency.png" width="620" alt="Quality vs on-device latency (BOOX SD662): the 25M ternary hits 76% 免選字 at ~9ms/6-syllable decode — same speed as a tiny 4M int8 but far higher quality, and both faster and better than the 12M int8 (13ms / 免選字 72)"></p>

A from-scratch **SlothE-T 25M ternary (W1.58A8) bidirectional encoder**. At on-device
latency (BOOX SD662) it is **Pareto-optimal**: **76% 免選字** at ~**9 ms / 6-syllable
decode** — the same speed as a tiny 4M int8 but far more accurate — and it is **both
faster and better** than the 12M int8 (13 ms, 免選字 72). **At this scale ternary beats
int8** (the opposite of the 11.6M). Architecture, GGUF, and the full reproduction pipeline:
[Luigi/slothe-t-25m-zhuyin](https://huggingface.co/Luigi/slothe-t-25m-zhuyin).
(\* 25M ternary latency projected from measured-shape TQ2_0 / I²_S kernels.)

## Numbers

Honest held-out (500 c4-zh-TW sentences, excluded from the training corpus):

| Benchmark | Score |
|---|---|
| 免選字 held-out (whole sentence exact) | **76%** (380/500) |
| Tonal per-char accuracy (homophone-hard) | **86%** (libchewing 71%) |
| Tone-free | **77%** |

Ceiling = 微軟新注音/自然輸入法; floor = libchewing. Methodology in
`docs/COMPARISON.md`. Speed: the TQ2_0 ternary kernel is ~2.3× int8 on x86
(mainline ggml kernel benchmark); no bitnet.cpp needed.

## Roadmap

- [x] ~10M model: 免選字 74→**84** (11.6M int8, old benchmark)
- [x] **25M ternary shipped to all four frontends**: 免選字 **76** / homophone
  **86** / tone-free **77** (honest held-out); all four share `libslothe`
  (ggml / TQ2_0) replacing ONNX Runtime — desktop daemon, Android APK (−20 MB),
  web WASM, all golden-validated. **At this scale ternary beats int8** (the
  opposite of 11.6M)
- [ ] **Char-hints v2**: retrain a hinted ternary model to restore pick /
  document-context re-scoring
- [ ] BIO word-boundary + model-based 聯想 head (needs a fine-tune); word-list filtering
- [ ] Android hardware-keyboard polish; regular desktop package releases
- [ ] **Senior-friendly keyboard layout**: standard layout + key-error-tolerant decoding; design research in [docs/SENIOR-KEYBOARD.md](docs/SENIOR-KEYBOARD.md)

<details><summary>Done (expand)</summary>

libchewing-free engine (keyboard FSM + LLM decode) · web demo · tone-free /
auto zh/en code-switch · SlothLM-E 11.6M (NAS-derived + g2pW) · char-hint channel
(pick re-scoring / document context / typo repair) · 新注音-style live
conversion + chewing-grade candidate window · differential UI-parity suite vs
real libchewing · full reproducibility bundle on HF · IBus engine · native
Android IME (validated on BOOX e-ink) · 聯想 on all four frontends · touch
candidate strip · learn-bonus calibration (2/3) · `.deb` / `.apk` packaging ·
**25M ternary model + libslothe (ggml/TQ2_0) deployed to all four frontends**
</details>

**Non-goals:** any cloud inference or telemetry — everything runs locally.
