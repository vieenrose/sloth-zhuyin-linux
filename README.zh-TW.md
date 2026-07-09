# sloth-zhuyin-linux · 懶 Slothing

**不依賴 libchewing、由 LLM 驅動的注音輸入法** — Linux 上的 fcitx5 輸入法,
外加一個完全在瀏覽器內執行的網頁 demo。打注音,一個從零訓練的小模型就把它
解碼成繁體中文,並受「音對得上」的文法約束——每個字都是你所打注音的真實讀音,
絕不亂猜。

**English: [README.md](README.md)**

> **線上 demo(免費、完全在你的瀏覽器內執行):**
> **https://huggingface.co/spaces/Luigi/slothing-web**
> 用畫面上的大千鍵盤或實體鍵盤打字——中文、英文、中英混打都會自動辨識,
> 不用切換模式。

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
| 參數量 | 約 34M | **約 16M** |
| 解碼 | 自回歸 | **非自回歸,一次前向** |
| 有聲調準確率 | 約勝過 chewing | **81%(chewing 71%)** |
| 免聲調準確率 | 弱 | **82%** — 真正可用的免聲調輸入 |
| HF | [Luigi/slothlm-34m-zhuyin](https://huggingface.co/Luigi/slothlm-34m-zhuyin) | ONNX 匯出中 |

注音解碼本質上是「對齊的序列標註」(N 個音節 → N 個字,一對一,各自受限),
所以**雙向編碼器**比因果解碼器更適合這個任務:它看得到整句(用右側語境
消歧:行走/銀行)、一次前向就解完、體積只有一半。兩者都在繁中語料 + 合成
任務(注音↔文字、免聲調、中英混打)上從零訓練,採用**自訂的位元組級
tokenizer**(每個注音符號/漢字一個 token)。詳見 `model/DESIGN.md` 與
`model/DESIGN-E.md`。

## 功能

- **文法約束解碼** — 輸出被限制在每個音節「音對得上」的字集,絕不會生出
  不合讀音的字。
- **免聲調輸入** — 省掉聲調鍵(約少打 35% 按鍵);模型靠語境消歧。
- **中英自動切換** — 不用切模式:合法注音每按一鍵只增加一個注音符號,所以
  一段「不可能是注音」的按鍵就被判為英文(華碩輸入法風格)。英文原樣輸出,
  中英混打(`我用 Python 寫 code`)直接可用。
- **貼近 chewing 的編輯體驗** — 邊打邊轉、Enter 直接上字、預編輯區游標與
  句中插入編輯、可翻頁的候選字加數字鍵選字、標點、逐字與 LLM 排序的
  詞候選、本次工作階段的學習記憶。
- **對照 libchewing 驗證** — 每次模型/解碼變更都必須先通過
  `eval/chewing_parity.py`(SlothLM ≥ chewing)才能發布。

## 專案結構

- `engine/fcitx5-chewing/` — **Slothing**,fcitx5 外掛。現已不依賴 libchewing:
  按鍵由 `src/zhuyin.h`(零相依的大千狀態機)解析;轉換鍵把打好的音節送到
  daemon 的解碼端。
- `engine/slothingd/` — **slothingd**,一個小型 Unix socket daemon(llama.cpp
  C API),在「合法字」GBNF 文法下解碼音節,支援有聲調/免聲調,並讓英文
  原樣通過以支援中英混打。
- `model/` — 各個模型:tokenizer、資料準備、SlothLM(解碼器)與 **SlothLM-E**
  (編碼器)的訓練與評測、`phonetic_table.tsv`(音節 → 合法字)、以及
  `chewing_parity.py`(驗證關卡)。
- `space-static/` — 網頁 demo(`sdk: static`):完整輸入法 UI,透過
  ONNX / Transformers.js 在瀏覽器內執行模型——免費、不會休眠、無伺服器。
- `eval/` — 有計分的注音→句子測試集與各評測工具(重排、解碼、chewing 對照)。
- `ARCHITECTURE.md`、`RESEARCH-LLM-IME.md`、`MODEL_BENCHMARKS.md`、`MIGRATION.md`。

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
sh scripts/setup-llm.sh          # llama.cpp + 模型 + slothingd
packaging/run-slothingd.sh       # 啟動解碼器(刻意手動啟動)
```

## 藍圖(重點)

- [x] 不依賴 libchewing 的引擎(鍵盤狀態機 + LLM 解碼)
- [x] SlothLM(34M 解碼器)已訓練、上 HF、附 Q4/Q8 GGUF
- [x] 網頁 demo — 瀏覽器內執行、免費、貼近 chewing 的操作
- [x] 免聲調模式、中英自動切換、中英混打、工作階段學習
- [x] SlothLM-E(16M 雙向編碼器)— 更小/更快/更準;收尾中
- [ ] 把 demo 與桌面 daemon 換成 SlothLM-E 的 ONNX 模型
- [ ] 逐詞下鍵重排;錯字容忍;打包(.deb)
- [ ] (未來,整篇文件語境)Transformer + SSM 混合解碼器

**非目標:** 任何雲端推論、遙測。一切都在本機執行。
