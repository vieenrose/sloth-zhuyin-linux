# sloth-zhuyin-linux

An LLM-enhanced Zhuyin (Bopomofo) input method for fcitx5 on Ubuntu/Kubuntu.

- `engine/fcitx5-chewing/` — **Slothing**, an fcitx5 input method addon forked
  from [fcitx5-chewing](https://github.com/fcitx/fcitx5-chewing) (pinned to
  5.0.14) and re-namespaced so it installs and runs side-by-side with the
  stock `fcitx5-chewing`/`fcitx5-zhuyin` packages. Uses libchewing for
  zhuyin-key → candidate generation.
- `engine/slothingd/` — **slothingd**, a small Unix-socket daemon linking
  directly against llama.cpp's C API (`llama.h`, no `llama-server`, no
  Python bindings) that reranks/corrects libchewing's candidate list at
  commit points using grammar-constrained decoding, fixing the classic
  homophone-disambiguation weakness of n-gram based engines.
- `llm/` — local LLM runtime: a llama.cpp checkout (built for its headers
  and shared libs, which `slothingd` links against) plus the GGUF model.
  Not vendored in git (see `.gitignore`); see setup instructions below.
- `RESEARCH.md` — notes on how LLMs are used to improve input methods
  elsewhere, gathered before designing the reranker.
- `MODEL_BENCHMARKS.md` — comparison of candidate reranker models; see
  `MODEL_BENCHMARKS.md` for why **LFM2.5-230M** was picked.

## Status

Milestone 1 (plain fcitx5 zhuyin engine, no LLM) is done. Milestone 2
(`slothingd` LLM reranker daemon) builds and works standalone; wiring it
into `eim.cpp`'s commit path is in progress.

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

hf download LiquidAI/LFM2.5-230M-GGUF --include "*Q8_0*" \
  --local-dir models/lfm2.5-230m
cd ..

cmake -B engine/slothingd/build -S engine/slothingd -DCMAKE_BUILD_TYPE=Release
cmake --build engine/slothingd/build -j"$(nproc)"
```

Run it with `packaging/run-slothingd.sh` (manual start; no auto-start/systemd
unit by design — run it yourself when you want the reranker active).
