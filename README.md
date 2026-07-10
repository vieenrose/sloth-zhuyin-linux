# sloth-zhuyin-linux · 懶 Slothing

**不依賴 libchewing、由 LLM 驅動的注音輸入法** — Linux(**fcitx5** 與
**IBus**)、**Android**(原生 IME,模型在裝置端跑),外加一個完全在瀏覽器內
執行的網頁 demo。打注音,一個從零訓練的小模型就把它解碼成繁體中文,並受
「音對得上」的文法約束——每個字都是你所打注音的真實讀音,絕不亂猜。

**English: [README.en.md](README.en.md)**

> **線上 demo(免費、完全在你的瀏覽器內執行):**
> **https://huggingface.co/spaces/Luigi/slothing-web**
> 用畫面上的大千鍵盤或實體鍵盤打字——中文、英文、中英混打都會自動辨識,
> 不用切換模式。

<p align="center"><img src="docs/demo-v8.gif" width="640" alt="Slothing 網頁 demo — 注音即打即解碼,中英混打"></p>

## 四種前端,同一個核心

四個前端都建立在 `engine/common` 的**前端無關核心**上(同一個狀態機、同一份
中英 DP 切分、同一個解碼實作),所以互動行為完全一致——上面的 GIF 是網頁
demo,但 fcitx5 / IBus 的操作一模一樣。

| 前端 | 平台 | 說明 |
|---|---|---|
| **fcitx5** | Linux(KDE/一般) | 原生外掛,不依賴 libchewing。`engine/fcitx5-chewing/` |
| **IBus** | Linux(GNOME 等) | 同核心;附無頭端對端測試(私有 ibus-daemon 逐鍵驗證)。`engine/ibus-slothing/` |
| **Android** | 手機/平板(含 e-ink) | 原生 Kotlin IME;把 slothingd 的解碼移植成裝置端 C++（ONNX Runtime,離線)。`android/` |
| **網頁 demo** | 瀏覽器 | onnxruntime-web;免安裝、免費、不會休眠。`space-static/` |

**Android 版**(在 ONYX BOOX Tab Mini C,e-ink 平板上實測)——原生大千注音
鍵盤、免選字整句轉換、完全離線。裝置端解碼與桌面同一顆模型:230 句免選字
基準 **172/230 = 74%**,與桌面原模型完全一致(逐句 99% 一致)。

與 Gboard 注音、Boox 內建輸入法的誠實對照(附來源):**[docs/COMPARISON.md](docs/COMPARISON.md)**

<p align="center"><img src="docs/android-boox-demo-v2.gif" width="460" alt="Slothing 在 BOOX Tab Mini C(Android)上:點大千注音鍵,免選字整句轉換出「我在寫程式」,候選列即打即現"></p>

## 這是什麼

Slothing 把傳統注音輸入法(如 chewing)的統計式解碼器換成一個小型語言模型,
同時保證輸出永遠是「音對得上」的合法字。有兩點讓它跟其他開源注音輸入法
(McBopomofo、vChewing、libchewing 都是純統計式)不同:

- **模型負責解碼,不只是重排。** 由注音→中文的模型,靠句子語境解決字典式
  輸入法常錯的同音字(它→他、在/再、覺/決)。
- **不用 libchewing。** 一個零相依的鍵盤狀態機負責解析按鍵;模型負責解碼;
  逐字的「合法字」文法保證讀音正確。本機執行、隱私、無雲端。

## 模型

| | SlothLM(v1) | **SlothLM-E**(v2) |
|---|---|---|
| 類型 | 因果解碼器 LM(Llama) | **雙向編碼器** |
| 參數量 | 約 34M | **3.8M(NAS＋權重共享)** |
| 解碼 | 自回歸 | **非自回歸,一次前向** |
| 有聲調準確率 | 約勝過 chewing | **83%(chewing 71%)** |
| 免聲調準確率 | 弱 | **70%** — 可用的免聲調輸入 |
| HF | (已移除) | [Luigi/slothlm-e-4m-zhuyin](https://huggingface.co/Luigi/slothlm-e-4m-zhuyin) |

注音解碼本質上是「對齊的序列標註」(N 個音節 → N 個字,一對一,各自受限),
所以**雙向編碼器**比因果解碼器更適合這個任務:它看得到整句(用右側語境
消歧:行走/銀行)、一次前向就解完。目前的模型只有 **3.8M 參數**,由 18 組
Hyperband **神經架構搜尋**(sub-5M 空間)找出,並以 **g2pW 語境感知讀音**
(神經式台灣破音字消歧)標註訓練。模型還帶有一條**字提示通道**(與輸出層
權重共享,幾乎零參數):使用者選字會回饋成提示,整句圍繞它**重新評分**
(微軟新注音式);同一通道也載入**文件語境**(游標前已上字的文字——
「我妹妹說」會讓 他很漂亮 變成 **她**很漂亮),並以**打字錯誤雜訊**訓練,
讓模型自己修正打錯的音節。完整重現流程(資料集 → 標註 → NAS → 訓練 →
ONNX)隨模型附在 HF。
詳見 `model/DESIGN.md` 與 `model/DESIGN-E.md`。

## 功能

- **文法約束解碼** — 輸出被限制在每個音節「音對得上」的字集,絕不會生出
  不合讀音的字。
- **免聲調輸入** — 省掉聲調鍵(約少打 35% 按鍵);模型靠語境消歧。
- **中英自動切換** — 不用切模式:合法注音每按一鍵只增加一個注音符號,所以
  一段「不可能是注音」的按鍵就被判為英文(華碩輸入法風格)。英文原樣輸出,
  中英混打(`我用 Python 寫 code`)直接可用。
- **貼近 chewing 的編輯體驗** — 邊打邊轉、Enter 直接上字、預編輯區游標與
  句中插入編輯、可翻頁且**依模型分數排序**的候選字(↓ 開啟、詞/單字視圖
  切換、數字鍵選字)、標點與 \` 符號選單、Shift 中英切換、Shift+空白
  全形/半形。
- **選字即重新評分** — 改一個字,整句其他字圍繞你的選擇重新解碼
  (字提示通道);選字也會被**持久學習**(校準過的 logit 加分,只翻
  近似同音字、不污染強語境字)。
- **打錯也能解** — 不合法的音節由模型從語境自動修正(編輯距離 1)。
- **UI 行為對照真正的 libchewing 驗證** — `eval/ui-parity/` 差分測試套件
  逐鍵比較 UI 狀態(12/12 個互動契約通過);模型品質另以
  `eval/chewing_parity.py` 與 230 句免選字測試把關。

## 專案結構

- `engine/common/` — **前端無關的共用核心**(單一實作、離線單元測試):
  中英 DP 切分 `segment.h`(與網頁 demo 同步)、注音鍵盤 FSM、slothingd
  協定客戶端、以及整個互動狀態機 `core.h`(候選窗、循環反白、選字重評分、
  學習差分)。fcitx5 與 IBus 兩個引擎都建立在它之上。
- `engine/fcitx5-chewing/` — **Slothing** fcitx5 外掛(共用核心的 fcitx5
  介接層)。不依賴 libchewing。
- `engine/ibus-slothing/` — **Slothing** IBus 引擎(GNOME 使用者適用),
  同一個核心、同樣的行為;附無頭端對端測試(私有 ibus-daemon 逐鍵驗證)。
- `android/` — **Slothing Android 版**:原生 Kotlin `InputMethodService` +
  大千注音鍵盤,建立在同一個 `engine/common` 核心上,並把 slothingd 的解碼
  移植成裝置端 C++(ONNX Runtime,經 JNI)——手機上完全離線、無 daemon。
- `engine/slothingd/` — 解碼 daemon:**`slothingd_e.py`**(現行;Unix socket
  的 onnxruntime daemon,服務 SlothLM-E,約 1 毫秒/次解碼,支援有聲調/
  免聲調、英文原樣通過),以及舊版 llama.cpp/GBNF 的 `slothingd.cpp`
  (GGUF 解碼器模型用)。
- `model/` — 各個模型:tokenizer、資料準備、SlothLM(解碼器)與 **SlothLM-E**
  (編碼器)的訓練與評測、`phonetic_table.tsv`(音節 → 合法字)、以及
  `chewing_parity.py`(驗證關卡)。
- `space-static/` — 網頁 demo(`sdk: static`):完整輸入法 UI,透過
  onnxruntime-web 在瀏覽器內執行模型(約 5 MB int8 ONNX,iOS Safari 也能跑)
  ——免費、不會休眠、無伺服器。
- `eval/` — 有計分的注音→句子測試集與各評測工具(重排、解碼、chewing 對照)。
- `ARCHITECTURE.md`、`RESEARCH-LLM-IME.md`、`MODEL_BENCHMARKS.md`、`MIGRATION.md`。

IBus 使用者(GNOME):見 `engine/ibus-slothing/README.md`(同一個核心、
同樣的按鍵行為)。Android 版:見下方與 `android/`。

## 建置與安裝 fcitx5 外掛

```sh
cmake -B engine/fcitx5-chewing/build -S engine/fcitx5-chewing \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr
cmake --build engine/fcitx5-chewing/build -j"$(nproc)"
sudo make -C engine/fcitx5-chewing/build install
fcitx5 -r -d
```

在 `fcitx5-configtool` 加入 **Slothing**(🦥)。接著設定本機模型 + daemon:

```sh
pip install onnxruntime numpy    # daemon 相依套件
hf download Luigi/slothlm-e-4m-zhuyin --local-dir model/slothe_4m_onnx \
    --include 'onnx/*' 'syl_vocab.json'   # 之後把 onnx/* 移到上一層
packaging/install-slothingd-service.sh   # 登入自動啟動(systemd 使用者服務)
# 或:packaging/run-slothingd.sh          # 一次性手動執行
```

## 建置與安裝 Android 版

需要 Android SDK/NDK(cmake、build-tools、platform）與 JDK 21。模型與 ONNX
Runtime 不進 git:先取模型、再抓 ORT(細節見 `android/.gitignore`)。

```sh
packaging/fetch-model.sh                 # 準備 model/slothe_4m_onnx(進 APK)
cd android
JAVA_HOME=/usr/lib/jvm/java-21-openjdk ANDROID_HOME=~/Android/Sdk \
  ./gradlew :app:assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
adb shell ime enable com.slothing.ime/.SlothingImeService
adb shell ime set    com.slothing.ime/.SlothingImeService
```

在裝置的輸入法設定啟用「Slothing 注音」即可(app 內也有一鍵入口)。手機上
完全離線、無 daemon——解碼在裝置端由 ONNX Runtime 執行。細節見
`android/README.md` 之外的 `engine/common/decoder.h`(共用解碼介面)。

## 藍圖(重點)

- [x] 不依賴 libchewing 的引擎(鍵盤狀態機 + LLM 解碼)
- [x] SlothLM(34M 解碼器)v1 — 已被 SlothLM-E 取代,自 HF 移除
- [x] 網頁 demo — 瀏覽器內執行、免費、貼近 chewing 的操作
- [x] 免聲調模式、中英自動切換、中英混打、工作階段學習
- [x] SlothLM-E 雙向編碼器;NAS 找出的 3.8M + g2pW 標註
- [x] 字提示通道:選字重新評分、文件語境、打錯自動修正(權重共享,零額外參數)
- [x] 微軟新注音式即打即轉(fcitx 免按轉換鍵)、chewing 級候選視窗行為
- [x] libchewing 差分 UI-parity 測試套件(行為可量測,不再逐案修)
- [x] demo 與桌面 daemon 都已換成 SlothLM-E 的 ONNX 模型(5 MB int8,無損)
- [x] HF 模型頁附上完整重現流程
- [x] 錯字容忍 — 模型評分的編輯距離 1 修正(demo + daemon)
- [x] IBus 引擎(GNOME):前端無關核心抽出共用,行為與 fcitx5 版一致
- [x] Android 原生 IME(第 4 個前端):共用核心 + 裝置端 ONNX 解碼,
  BOOX e-ink 實測(74% 免選字,與桌面模型 99% 逐句一致)
- [ ] 逐詞下鍵重排;打包(.deb)
- [ ] (未來,整篇文件語境)Transformer + SSM 混合解碼器

**非目標:** 任何雲端推論、遙測。一切都在本機執行。
