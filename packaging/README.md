# Packaging

## fcitx5 addon .deb

The addon (the part that needs a system install) builds into a Debian package
via CPack. The LLM daemon and model stay user-local (see
`scripts/setup-llm.sh`) because llama.cpp isn't a system package — and the
addon degrades gracefully to plain chewing behaviour when the daemon isn't
running, so the .deb is useful on its own.

```sh
cmake -B build -S engine/fcitx5-chewing -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr
cmake --build build -j"$(nproc)"
( cd build && cpack -G DEB )
sudo apt install ./build/fcitx5-sloth_*_amd64.deb
```

Produces `fcitx5-sloth_<version>_<arch>.deb`. Runtime deps
(`fcitx5`, `libchewing3`, the `libfcitx5*` libraries, libc/libstdc++) are
resolved automatically via `dpkg-shlibdeps`; `fcitx5-chinese-addons` is
recommended.

After install, restart fcitx5 and add **樹懶智慧輸入法** (Sloth IME) via
`fcitx5-configtool`, then run `packaging/run-slothd.sh` to enable LLM
conversion (Ctrl+Enter).

## run-slothd.sh

Manual launcher for the reranker daemon (see the script header). Started by
hand on purpose — no systemd unit / auto-start.

## Sloth IME theme (optional)

Sloth-brown fcitx5 candidate-panel themes (light + dark) matching the 懶
icon, under `packaging/themes/`:

```sh
packaging/install-theme.sh --set   # install both + activate the light one
fcitx5 -r -d
```

Or pick "Sloth IME" / "Sloth IME Dark" in fcitx5-configtool → Global Options →
Theme. Per-user, no sudo.
