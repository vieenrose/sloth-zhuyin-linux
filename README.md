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
  whole sentence as usual, then press the **convert key** (default **F9**) to
  get the LLM's alternatives in a normal fcitx5 candidate list — number keys /
  arrows select, Enter/Space commits the highlighted one, Esc cancels. Nothing
  runs during ordinary typing, and the LLM never rewrites text on its own; you
  always pick from the list. Accepting a candidate also teaches libchewing the
  correction (via `chewing_userphrase_add`) so it converges over time. The
  daemon binds a per-user socket under `$XDG_RUNTIME_DIR`.
- `llm/` — local LLM runtime: a llama.cpp checkout (built for its headers
  and shared libs, which both `slothingd` and the engine link against) plus
  the GGUF model. Not vendored in git (see `.gitignore`); see setup below.
- `RESEARCH.md` — notes on how LLMs are used to improve input methods
  elsewhere, gathered before designing the reranker.
- `MODEL_BENCHMARKS.md` — comparison of candidate reranker models; see
  `MODEL_BENCHMARKS.md` for why **LFM2.5-230M (Q4_0 quant)** was picked.

## Status

Milestone 1 (plain fcitx5 zhuyin engine, no LLM) is done. Milestone 2 (LLM
conversion) was hardened after an adversarial code review found a cluster of
security/lifetime/UX bugs in the first async-suggestion design; it is now a
**pull-model conversion** (convert key → native candidate list) with a
hardened daemon (per-user socket, SIGPIPE-safe, read-timeout) and a
lifetime-safe background worker. Example: type ㄨㄛˇㄗㄞˋㄔㄨㄥˊㄒㄧㄣㄎㄠˇㄌㄩˋ,
press F9 → 我在重新考慮 offered where raw chewing produces 我再重新考慮.

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

Run it with `packaging/run-slothingd.sh` (manual start; no auto-start/systemd
unit by design — run it yourself when you want the reranker active).
