# Deep research: LLMs in Chinese input methods (Taiwan & China)

Multi-source, adversarially verified research pass (July 2026): 5 search
angles, 24 sources fetched, 115 claims extracted, 25 surviving 3-vote
verification, 0 refuted. Emphasis: concrete techniques and shipped-vs-research
status, to position Sloth IME.

## China (pinyin ecosystem): the research→industry arc

**PinyinGPT — Tencent AI Lab, ACL 2022** ([arXiv:2203.00249](https://arxiv.org/abs/2203.00249))
First published use of a Chinese GPT for pinyin conversion. A *frozen* GPT
under constrained decoding reaches SOTA on perfect (fully spelled) pinyin —
but collapses on abbreviated (initials-only) input. Two fixes recover it:
pinyin-enriched context and pinyin-constrained training to separate
homophones. *Validates Sloth IME's core premise (generic LM + phonetic
constraints beats traditional conversion) and flags abbreviated/partial input
as the hard case.*

**GeneInput — iFlytek + USTC, EMNLP 2023 Industry** ([arXiv:2311.01166](https://arxiv.org/pdf/2311.01166))
The most complete commercial LLM-IME design published:
- Generative "Full-mode Key-sequence to Characters": 26-key perfect &
  abbreviated, 9-key, mixed, and *noisy* input (extra/missing/reversed/wrong
  keys) — drops the assumption that the user typed correct pinyin at all.
- **2.6B-param** Spark-based model (32-layer GPT-like), size explicitly chosen
  against the cost of hundreds of millions of daily IME calls. **Cloud-side**;
  on-device shrinking was named as future work (Nov 2023).
- Results: P@1 = 88.4 / P@5 = 96.2 (People's Daily) vs PinyinGPT 73.2 and an
  online production P2C baseline 71.3 — a ~15-17 point quantification of what
  the LLM adds. (Self-reported; baselines copied, not re-run.)
- **Personalization via implicit RLHF**: reward models (Chinese
  DeBERTa-v2-large) trained on real users' candidate selections —
  selected = positive, shown-but-unselected = negative. No manual annotation.

**PERT** ([arXiv:2205.11737](https://arxiv.org/abs/2205.11737))
BERT-style bidirectional encoder for P2C, beating n-gram/RNN baselines —
deployed as a **hybrid**: combined with n-gram models under a Markov
framework, plus an external lexicon as the escape hatch for
out-of-distribution words (names, jargon).

**PY-GEC — Huawei** ([arXiv:2409.13262](https://arxiv.org/html/2409.13262v1))
Off-the-shelf LLaMA-3-8B-Chinese aligns pinyin and text representations
poorly (cos 0.26); multitask fine-tuning with pinyin↔text conversion tasks
raises alignment to 0.82 and improves downstream correction (CER 11.48→10.53,
entity recall 70.19→72.93). *(n=1, ASR domain — extrapolation to zhuyin is
plausible but inferred.)*

**Li et al. 2014 (pre-neural reranking)** — generate k-best via n-gram beam
search, linear-rerank with tuned feature weights: CER 12.03→10.94. The key
number: **500-best oracle CER = 2.01 vs 11.27 for 1-best** — the right answer
is almost always *in* the candidate set; ranking is the bottleneck.
*(Low-prestige venue, minor internal arithmetic inconsistency; the oracle
observation is the durable insight.)*

## Taiwan (zhuyin ecosystem): no LLM anywhere in open source

Verified against current repos/wikis (June–July 2026):
- **McBopomofo** (+ the official fcitx5 Linux port): Gramambular grid-builder,
  a lattice/Viterbi highest-probability search over a **unigram-only** LM
  (wiki-confirmed — no bigrams, let alone neural).
- **vChewing**: frequency-based Homa engine, separate TC/SC dictionaries —
  traditional statistical.
- **libchewing / fcitx5-chewing**: classic n-gram/HMM, as Sloth IME already
  knows firsthand.

**Sloth IME's local-LLM grammar-constrained reranking has no verified
open-source precedent in the zhuyin ecosystem.** Even iFlytek's commercial
system couldn't yet run on-device at 2.6B in 2023 — a local 230M model on
desktop Linux is precisely the gap they named as future work.

## Apple Intelligence & zh-TW (follow-up pass, July 2026)

- **Apple Intelligence gained Traditional Chinese in iOS 26.1 (Nov 2025)**
  ([Apple TW newsroom](https://www.apple.com/tw/newsroom/2025/11/apple-intelligence-features-are-now-available-in-traditional-chinese/),
  [TechNews](https://technews.tw/2025/09/16/apple-intelligence-traditional-chinese/)):
  Writing Tools (rewrite/summarize), Live Translation, Image Playground,
  photo search, Siri — on iPhone 15 Pro+ in the 中文（國語－台灣） locale.
  Crucially, these operate on text *after* it is typed. No evidence surfaced
  that Apple Intelligence improves the **zhuyin conversion/選字 step itself**.
- **iOS 17's transformer autocorrect** (the "transformer language model"
  keyboard feature, [MacRumors](https://www.macrumors.com/guide/ios-17-keyboard/))
  was English-centric; no verified zhuyin conversion improvement.
- **Keyly** ([keylyapp.com](https://keylyapp.com/en/)) — a third-party
  commercial **AI zhuyin keyboard for iPhone/iPad**: offline-capable
  (privacy-first) with optional cloud models, AI rewrite/translate inside the
  typing flow. Validates the "AI zhuyin, local-first" product thesis
  commercially; closed-source, iOS-only.

**Positioning takeaway:** even Apple's zh-TW AI push is post-typing writing
assistance layered *around* the IME — the conversion step Taiwanese users
fight daily (homophone 選字) remains unimproved by Apple. Sloth IME attacks
exactly that step, on desktop Linux, open source; Keyly's existence shows
there is commercial appetite for local-first AI zhuyin, with no open-source
or Linux equivalent.

## Coverage gaps (absence of evidence ≠ evidence of absence)
No claims survived (or were found) on: Sogou/Baidu/WeChat-keyboard *shipped*
LLM features; ASUS 智慧輸入法 internals; Apple/Google zh-TW; RIME/librime LLM
plugins; PTT/v2ex community experiments. Also unknown: whether iFlytek shipped
on-device GeneInput after Nov 2023.

## Implications for Sloth IME
1. **Reranking headroom is proven** (oracle CER 2.01 vs 11.27) — improving
   selection over libchewing's lattice is the right lever before considering
   generative decoding.
2. **Fine-tune with zhuyin↔text tasks**: a generic 230M base likely has weak
   zhuyin-text alignment (PY-GEC's 0.26→0.82 lesson); explicit conversion
   tasks in fine-tuning should materially lift reranking quality.
3. **Implicit feedback is free training signal** (GeneInput's RLHF recipe):
   Sloth IME's accepted/rejected conversion candidates could drive local
   preference tuning with zero privacy cost — all on-device.
4. **Noisy/abbreviated input is the next frontier**: GeneInput's FK2C scope
   (wrong/missing keys, initials-only) is where generation beats reranking;
   Sloth IME's tone-free-zhuyin ambition maps onto this.
5. **Hybrid is the shipped pattern everywhere** (PERT + n-gram + lexicon;
   GeneInput atop existing infra): keeping libchewing as the hot path matches
   industry practice, not just pragmatism.
6. **Positioning**: "first open-source zhuyin IME with local-LLM conversion"
   is defensible against everything verified here; the differentiators are
   *local/private* (vs iFlytek's cloud 2.6B) and *grammar-constrained safety*
   (output provably built from real candidates).
