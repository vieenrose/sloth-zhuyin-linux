# engine/common — the shared, frontend-free IME core

One implementation of Slothing's chewing-parity behavior, used verbatim by
both the fcitx5 addon (`engine/fcitx5-chewing`) and the IBus engine
(`engine/ibus-slothing`). Everything here is header-only, has no
fcitx/ibus/glib dependency, and is unit-tested offline:

| header | contents | tests |
|---|---|---|
| `zhuyin.h` | Dàqiān key→bopomofo tables, ZhuyinBuffer FSM | `../fcitx5-chewing/src/zhuyin_test.cpp`, `zhuyin_keys_test.cpp` |
| `segment.h` | DP keystream segmenter (auto zh/en), lock-step with `space-static/segment.js` | `../fcitx5-chewing/src/segment_test.cpp` (mirrors `test-segment.mjs`) |
| `daemon.h` | slothingd Unix-socket JSON client (decode / hints / phrases / learn) | exercised end-to-end by the IBus smoke test |
| `display.h` | UTF-8 utils, zh/en join spacing, fullwidth, symbol menu, punctuation map | `core_test.cpp` |
| `core.h` | ComposingCore (token buffer + cursor + stale-preserving display) and ChoosingCore (candidate window: highlight loop, pick-closes-window, hint re-scoring, learn diff) | `core_test.cpp` |

Run the core tests:

```bash
cd engine/common
g++ -std=c++17 -I . core_test.cpp -o /tmp/ct -pthread && /tmp/ct
```

`nlohmann/json.hpp` is vendored here (v3.11.3) so both engines build without
the llama.cpp checkout.

Frontends own only: key-event decoding into core calls, async decode workers
(std::thread + event-loop dispatch), and painting. When changing interaction
behavior, change it HERE, add a contract to `core_test.cpp`, and re-run the
UI-parity suite (`eval/ui-parity`) — both engines pick the change up on
rebuild.
