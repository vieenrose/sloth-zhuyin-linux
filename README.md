# 懶音輸入法(Slothing)— LLM 注音輸入法

**打注音,模型整句轉換。** 懶音輸入法(全名「樹懶注音輸入法」,英文名 Slothing)
用一顆從零訓練的 11.6M 語言模型在本機把注音解碼成
繁體中文——不依賴 libchewing,每個字保證「音對得上」,絕不亂猜。桌面
(fcitx5、IBus)、Android、瀏覽器四個前端,共用同一個核心與同一顆模型。

**English: [README.en.md](README.en.md)** ·
**線上試用(免安裝):[huggingface.co/spaces/Luigi/slothing-web](https://huggingface.co/spaces/Luigi/slothing-web)**

<p align="center"><img src="docs/demo-web-v11.gif" width="470" alt="網頁 demo(BOOX 平板實錄):打「我用claude寫注音輸入法」——即打即轉、中英自動切換、點字改字+整句重評分、上字後聯想"></p>
<p align="center"><img src="docs/android-boox-demo-v8.gif" width="460" alt="Android 原生輸入法(BOOX e-ink 實錄):大千鍵盤、免選字、點字改字+整句重評分、聯想接龍;實體鍵盤按鍵即時回顯"></p>

## 特色

| | |
|---|---|
| **免選字整句轉換** | 微軟新注音式即打即轉;230 句免選字基準 **84%**(裝置端 83%) |
| **中英自動切換** | 不用切模式:`我用python寫程式` 直接打,DP 切分器自動判斷 |
| **免聲調** | 省掉聲調鍵(少打約 35%),模型靠語境消歧 |
| **選字即重評分** | 改一個字,整句圍繞你的選擇重新解碼(字提示通道),並**持久學習** |
| **打錯也能解** | 不合法音節由模型自動修正(編輯距離 1) |
| **聯想** | 上字後預測下一個詞(詞典＋個人習慣):行動點選接龍、桌面 ⇧1-9 |
| **完全離線** | 13 MB int8 ONNX 模型本機執行,零雲端、零遙測 |

與 Gboard 注音、Boox 內建輸入法的誠實對照(附來源):**[docs/COMPARISON.md](docs/COMPARISON.md)**;
四前端 UI 邏輯對照:**[docs/UI-MATRIX.md](docs/UI-MATRIX.md)**。

## 安裝

桌面平台需要解碼 daemon(一次設定,fcitx5 與 IBus 共用):

```sh
pip install onnxruntime numpy
packaging/fetch-model.sh                  # 下載 4.9 MB 模型
packaging/install-slothingd-service.sh    # 登入自動啟動
```

| 平台 | 安裝 |
|---|---|
| **fcitx5**(KDE 等) | Releases 的 `.deb`,或 `cmake -B engine/fcitx5-chewing/build -S engine/fcitx5-chewing -DCMAKE_INSTALL_PREFIX=/usr && cmake --build engine/fcitx5-chewing/build -j$(nproc) && sudo make -C engine/fcitx5-chewing/build install` |
| **IBus**(GNOME 等) | Releases 的 `.deb`,或一鍵腳本 `engine/ibus-slothing/install.sh`;詳見 `engine/ibus-slothing/README.md` |
| **Android** | Releases 的 `.apk`(手機上**免 daemon**,解碼在裝置端),或 `cd android && ./gradlew :app:assembleDebug`(需 SDK/NDK,模型先 `packaging/fetch-model.sh`) |
| **瀏覽器** | 免安裝:[HF Space](https://huggingface.co/spaces/Luigi/slothing-web) |

## 它怎麼運作

注音→中文是「對齊的序列標註」(N 音節 → N 字,各自受限於同音字集),所以用
**雙向編碼器**(非自回歸,一次前向)而非因果 LM:11.6M 參數(由 sub-5M
Hyperband NAS 冠軍配方放大)、g2pW 語境讀音標註訓練、**字提示通道**(與輸出層權重共享,
近零參數)承載選字回饋、文件語境與錯字修復。鍵流由零相依的 DP 切分器解析
(中英自動判斷),解碼輸出逐字限制在合法讀音內。

四個前端(fcitx5 / IBus / Android / 網頁)都是 `engine/common` 共用核心的薄
介接層——同一個狀態機、同一份切分器、同一個聯想引擎;行為以離線契約測試
(core_test)、IBus 無頭端對端測試與 `eval/ui-parity` 差分套件把關。

- 模型與完整重現流程(資料→標註→NAS→訓練→ONNX):[Luigi/slothlm-e-4m-zhuyin](https://huggingface.co/Luigi/slothlm-e-4m-zhuyin)
- 架構與設計:`ARCHITECTURE.md`、`model/DESIGN-E.md`、`MODEL_BENCHMARKS.md`

## 數字

| 基準 | 分數 |
|---|---|
| 230 句免選字(整句全對)| **84%**(193/230;Android 裝置端 83%)|
| 有聲調逐字準確率 | **83%**(libchewing 71%)|
| 免聲調 | 70% |

天花板 = 微軟新注音/自然輸入法(該測試集即其免選字答案集);樓板 =
libchewing。量法與對照見 `docs/COMPARISON.md`。

## 藍圖

- [x] ~10M 模型:免選字 74→**84**(11.6M,已上線四前端;int8 勝 int4/QAT/ternary — 見 experiment ledger)
- [ ] BIO 詞界＋模型式聯想頭(需微調);詞表交集過濾非詞
- [ ] Android 實體鍵盤完善;桌面套件常態發佈
- [ ] **銀髮族鍵盤佈局**(Android):3×4 大九宮格+模型消歧;設計研究見 [docs/SENIOR-KEYBOARD.md](docs/SENIOR-KEYBOARD.md)

<details><summary>已完成(展開)</summary>

libchewing-free 引擎(鍵盤 FSM＋LLM 解碼)· 網頁 demo · 免聲調/中英混打 ·
SlothLM-E 11.6M(NAS 衍生＋g2pW)· 字提示通道(選字重評分/文件語境/錯字修復)·
新注音式即打即轉＋酷音級候選窗 · libchewing 差分 UI-parity 套件 · HF 完整
重現流程 · IBus 引擎 · Android 原生 IME(BOOX e-ink 實測)· 四前端聯想 ·
觸控候選列 · 學習加分校準(2/3)· `.deb` / `.apk` 打包框架
</details>

**非目標:** 任何雲端推論、遙測——一切都在本機執行。
