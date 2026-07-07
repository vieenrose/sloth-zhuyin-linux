# sloth-zhuyin-linux

An LLM-enhanced Zhuyin (Bopomofo) input method for fcitx5 on Ubuntu/Kubuntu.

- `engine/fcitx5-chewing/` — **Slothing**, an fcitx5 input method addon forked
  from [fcitx5-chewing](https://github.com/fcitx/fcitx5-chewing) (pinned to
  5.0.14) and re-namespaced so it installs and runs side-by-side with the
  stock `fcitx5-chewing`/`fcitx5-zhuyin` packages. Uses libchewing for
  zhuyin-key → candidate generation.
- `llm/` — local LLM runtime (llama.cpp + a small Qwen2.5 GGUF model) used to
  rescore/correct full-sentence candidates at commit points, fixing the
  classic homophone-disambiguation weakness of n-gram based engines. Not
  vendored in git (see `.gitignore`); see setup instructions below.
- `RESEARCH.md` — notes on how LLMs are used to improve input methods
  elsewhere, gathered before designing the reranker.

## Status

Milestone 1 (plain fcitx5 zhuyin engine, no LLM) is done. Milestone 2 (LLM
reranker daemon) is in progress.

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

## Set up the local LLM runtime

```sh
mkdir -p llm && cd llm
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
cmake -B llama.cpp/build -S llama.cpp -DCMAKE_BUILD_TYPE=Release
cmake --build llama.cpp/build -j"$(nproc)" --target llama-server llama-cli

hf download Qwen/Qwen2.5-0.5B-Instruct-GGUF --include "*q4_k_m*" \
  --local-dir models/qwen2.5-0.5b
```
