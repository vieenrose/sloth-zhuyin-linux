# ibus-slothing

Slothing 的 IBus 引擎 — 與 fcitx5 版共用同一個前端無關核心
（`engine/common/core.h`），因此互動行為完全一致（酷音/新注音使用者零學習
成本）：

- 微軟新注音式即時整句轉換（免選字），穩定顯示、不會回退成注音符號
- 自動中英混打（DP 鍵流切分，與 fcitx5 / 網頁版同一份 `segment.h`）
- ↓ 開啟游標字候選窗（模型排序），詞候選在輔助列（⇧1-9 或 ←→+⏎）
- 選字後以字提示（hint channel）重新評分整句、更新其餘候選
- 一鍵 Enter 上字；單獨 Shift 切英文直通模式（微軟式 passthrough）
- ` 符號選單、Shift+Space 全形、酷音式 Esc / Backspace / 失焦上字語意
- 個人化學習（與 fcitx5 版共用 slothingd 的 learn.tsv）

## 建置與安裝

```bash
sudo apt install libibus-1.0-dev cmake g++   # Debian/Ubuntu
cmake -S engine/ibus-slothing -B engine/ibus-slothing/build \
      -DCMAKE_INSTALL_PREFIX=/usr
cmake --build engine/ibus-slothing/build -j$(nproc)
sudo cmake --install engine/ibus-slothing/build
ibus restart
ibus engine slothing        # 或在系統設定的輸入來源加入「Slothing 注音」
```

需要 slothingd（與 fcitx5 版同一隻守護程式）：見
`packaging/install-slothingd-service.sh`。

## 測試

無頭端對端測試（私有 dbus + ibus-daemon + 真實引擎 + 客戶端敲鍵，
需 slothingd 執行中）：

```bash
# 建好 engine 與 smoke 後
ENGINE=path/to/ibus-engine-slothing SMOKE=path/to/smoke \
  engine/ibus-slothing/test/run-smoke.sh
```

狀態機本身的酷音行為契約在 `engine/common/core_test.cpp`（離線、免
daemon），fcitx5 與 IBus 兩個前端共用同一份實作。

## GNOME 注意事項

GNOME Shell 自行繪製候選窗與輔助列；詞候選列（⇧1-9）顯示在輔助文字
內，樣式由 Shell 決定。行為（按鍵、循環反白、選後關窗）與 fcitx5 版相同。
