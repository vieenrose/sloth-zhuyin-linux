<!-- Generated 2026-07-14 via the gpt-ime-design workflow (synthesis of two independent designs:
     pure one-LLM maximalist + pragmatic hybrid). Companion: docs/IOS-PARITY-MOBILE.md. -->

# GPT-IME: architecture, dataset & training

*Authoritative design (deliverable #2). Merges Design A (pure one-LLM maximalist) and Design B (pragmatic hybrid) into one buildable plan that satisfies every iOS-parity requirement via the inside-constrained / outside-free decoding paradigm.*

## Summary & recommendation

One autoregressive Chinese+English LM owns the entire IME surface — full-sentence conversion, n‑best, char/word/phrase candidates, emoji, next‑word association, continuation, and zh‑EN code‑switch — produced in **one decode over one model** where the **decode position relative to the typed span is the only task switch** (inside the span → phonetic‑constrained conversion; outside → free generation). We adopt the maximalist thesis (Design A) but ship it on the pragmatic base and latency ramp (Design B), resolving the two designs' disagreements as follows:

- **Base model: SmolLM‑Chinese‑180M ships; Qwen2.5‑0.5B is the permanent teacher / flagship upgrade rung.** The user ceiling is ~250M full precision; SmolLM‑180M is the only cached pretrained base under that ceiling, is warm‑startable, and is already the default in `gen_convert.py`. Qwen2.5‑0.5B (494M) is over budget, so it is *not* the ship target (Design A's weak point) but is retained as the accuracy‑ceiling teacher for distillation and as an optional flagship‑only model (Design B's stance, kept).
- **Input serialization: whole‑syllable tokens** (Design B) over atomic bopomofo symbols (Design A) — 1 input token per syllable gives exact 1:1 input‑syllable ↔ output‑Han alignment (the pointer *is* the output‑char count), compact prefill, and a strong per‑syllable warm‑start (mean of the syllable's legal chars, the `bert_convert.py --warmstart` +3pt trick).
- **Decode: KV‑cached constrained beam** inside the span (phonetic + tone‑union mask reused verbatim from `slothd_e.py`), free continuation outside; **one length‑normalized log‑prob scale** unifies char/word/phrase/sentence/emoji/prediction into the labelless iOS candidate stream.
- **Latency ramp (accuracy first): bf16 → GGUF Q4_K → encoder‑as‑drafter speculative decoding → distillation.** The shipping 25M ternary SlothE encoder becomes the speculative *draft* proposer (never the answer); the GPT verifies. This is the shared centerpiece of both designs.
- **Drop‑in behind the existing `decoder.h Decoder` seam** (`decode`/`decodeWithHints`/`phrasesScored`/`learn`) with zero UI change; segmenter, `assoc.h`, learn store, and `core.h ChoosingCore` are unchanged.

Why this over the alternative: Design A's Qwen‑0.5B‑as‑ship bet violates the explicit ≤250M budget and rests on a distillation‑recovers‑quality gamble; Design B's SmolLM‑180M respects the budget today. But Design B under‑commits to the single‑model ambition (it hedges on the encoder). We take A's ambition (one model owns everything, outside‑span free generation is native prediction/emoji) with B's budget discipline and hybrid latency hedge — and keep Qwen‑0.5B as the teacher so the 180M student inherits the ceiling A was chasing.

---

## Architecture

### Base model

| Role | Model | Params | Precision / footprint | Why |
|---|---|---|---|---|
| **Ship v1** | **SmolLM‑Chinese‑180M** | 180M | bf16 ≈ 360 MB → Q4_K ≈ 110 MB → ternary (libslothe) ≈ 60–70 MB | Only cached pretrained base ≤250M; Han+Latin tokenizer; `gen_convert.py` default; warm‑startable; fastest to phone. |
| **Teacher / ceiling** | Qwen2.5‑0.5B | 494M | bf16 ≈ 1.0 GB → Q4_K_M ≈ 350 MB | Strongest TW‑zh + English + native BPE emoji tokens. Defines the accuracy ceiling, is the KD teacher, and is the **flagship‑only** upgrade rung (identical pipeline/seam). Over the ≤250M budget, so never the default ship model. |
| Research ceiling | Qwen2.5‑1.5B | 1.5B | — | Out of phone budget; used only to sanity‑check how much headroom remains. |

Warm‑start lesson from the encoder bake‑off is binding: *ranking accuracy tracks LM‑pretraining quality; from‑scratch scaling hurt toneless (136M cold → 74.2 vs 25M baseline 81); every warm‑start beat baseline (roberta‑base+warmstart 86.8 toned / 84.9 toneless).* We therefore **never cold‑train**; the 180M student is additionally KD‑supervised by the 0.5B teacher so it inherits quality the encoder can't reach on toneless.

### Input serialization — whole‑syllable tokens (resolved)

`phonetic_table.tsv` has **1474 toned surface rows** over **417 distinct tone‑stripped bases**. We add exactly those **1474 surface forms as new special tokens** (`⟨s:ㄓㄜ⟩ ⟨s:ㄕˋ⟩ ⟨s:ㄕㄨ⟩ …`), materialized as `syl_vocab.json` (the file `gen_convert.py`/`eval_arch.py` already reference by that name — build it from the table keys). Rationale for choosing this over Design A's 39 atomic bopomofo + 4 tone tokens:

1. **Exact 1:1 alignment.** One syllable = one input token, so the runtime pointer `p` (output‑Han count) indexes the input directly and the phonetic mask is a single lookup `legal[p]`. Atomic bopomofo makes a syllable span 1–3 tokens, forcing the pointer to count `<syl>` delimiters and lengthening prefill 2–4×.
2. **Strong warm‑start (the +3pt lever).** Each syllable‑token embedding is initialized as the **mean of its legal candidate chars' pretrained input embeddings** — literally `bert_convert.py --warmstart` (`from eval_arch import tone_union_chars`). Atomic‑symbol warm‑start (mean of all chars whose reading *starts* with a symbol) is far more diffuse.
3. **Closed inventory, no OOV.** Mandarin syllables are a closed set of 1474; there are no novel syllables to generalize to, so Design A's atomicity advantage does not apply. +1474 embedding rows is negligible.

Tone‑1 vs toneless: their surface strings are identical (both `ㄅㄚ`), so they map to **one** token; disambiguation is a decode‑time mask+prior, not a token distinction (see Decoding → toneless). English/number runs stay as **native BPE subwords** wrapped in `⟨en⟩…⟨/en⟩` sentinels marking a literal‑copy span. Add three control tokens: `⟨sep⟩` (span boundary, replacing `gen_convert.py`'s `" →"`; tokenizer‑agnostic so it works identically for SmolLM and Qwen), `⟨en⟩`, `⟨/en⟩`.

**Han output coverage guarantee.** Output stays native‑vocab, restricted to **length‑1 Han tokens** (`gen_convert.py`'s `id2char = {len(k)==1}` filter) so 1 output token = 1 char = 1 pointer advance. Audit every char in `phonetic_table.tsv` against the base tokenizer; any char lacking a single‑token id is **added as a new single‑char token** (warm‑started from its byte‑composition mean) so the 1:1 invariant holds universally. Design A's trie‑constrained sub‑emission is kept only as a last‑resort safety net for anything missed. Because the new syllable tokens must never be *generated*, hard‑mask all `⟨s:*⟩` ids out of every output step (and untie their rows from the output projection if the base ties embeddings — verify per base).

### Template & inside/outside boundary

```
⟨bos⟩ ⟨s:ㄓㄜ⟩ ⟨s:ㄕˋ⟩ ⟨s:ㄕㄨ⟩ ⟨s:ㄌㄢˇ⟩ ⟨en⟩world cup⟨/en⟩ ⟨sep⟩ 這 是 樹 懶 world cup ⟨eos⟩
      └──────────────── INPUT span (syllable tokens + literal EN) ─────────────┘        └────────── OUTPUT span ──────────┘
```

The **inside/outside boundary is a positional counter, not a token the model predicts** (both designs converge here — the clean choice). Maintain pointer `p ∈ [0, N_syl]`, `N_syl` = number of typed syllable tokens. Generation starts after `⟨sep⟩` with `p=0`. Each emitted length‑1 Han token advances `p` by 1 (enforced). Each `⟨en⟩…⟨/en⟩` literal run is **force‑copied** from the input and does **not** advance `p` (English is context, not a syllable slot). While `p < N_syl` → **inside**, mask on. When `p == N_syl` → **outside**, mask off, free generation. No boundary token to hallucinate.

This is exactly today's JSONL (`gen_data.py` `{"in":…, "out":…}`) rendered through `gen_convert.py`, with `" →"` → `⟨sep⟩`, syllable substrings → `⟨s:*⟩` tokens, and English wrapped in `⟨en⟩…⟨/en⟩`. **No new data‑generation code is required for the core pairs** — `gen_data.py` already emits toned/toneless/partial variants with code‑switch and emoji preserved verbatim.

---

## Decoding

Precompute once at load, reusing the **identical** legality builder as `slothd_e.py:83–137` / `eval_arch.py:19–37`:

- `legal[s]` = set of output Han token‑ids for syllable token `s`. Toned surface row → that row's chars (`tonal[s]`). Unmarked/tone‑1 row → **tone‑union**: union over all rows sharing the tone‑stripped base (`toneless[base]`, the `cands()` path). Impossible base (key‑slip) → edit‑distance‑1 `typo_fixes` union. Stored as `[n_syl][⌈n_char/64⌉]` uint64 bitsets; at a constrained step, `logits[illegal] = -inf`.

**Prefill once:** run `⟨bos⟩ … ⟨sep⟩` through the model, keep the KV‑cache. Everything below reuses it (the entire point of the AR redesign).

### Inside the span (constrained)

**(a) Best sentence.** Incremental greedy over the output span: at pointer `p`, one KV‑cached forward → mask `legal[p]` → `argmax` → append → advance. If `p` sits on an `⟨en⟩` run, force‑copy the literal English tokens (zero search, zero hallucination), don't advance `p`. `N_syl` steps → the boxed lead conversion (req. #3). Unlike the encoder's per‑position argmax (`slothd_e.py:245–256`), each char is conditioned on the **actual committed prefix** → true joint `P(懶|樹)·P(樹)`, not independent marginals.

**(b) n‑best sentences — constrained beam.** Beam width **B=8** (accuracy‑first; tunable down for latency; A's B=5 = `kLiveNBest` is the fast preset). Because alignment is strictly 1:1, **all beams advance `p` in lockstep** and share length at every step, so cumulative masked log‑probs are directly comparable *within the beam* — no length normalization needed for n‑best sentence ranking. English runs are force‑copied identically across beams (constant offset, no perturbation). Returns top‑k sentences → `getLiveSuggestions` / `liveSents_` (`session.h:387–391`), replacing the fake single‑flip n‑best (`slothd_e.py:252–255`) with a real beam.

```
beams = [(score=0, ids=[], p=0, kv=prompt_kv)]
for step in range(N_syl):
    if input[p] is EN-run: force-copy literal to every beam; continue   # no p advance
    for beam: expand only over legal[input[p]]; score += log_softmax(masked_logits)[tok]
    keep top-B by cumulative score; advance each beam's p
top-B → n-best sentences (beam[0] = headline)
```

**(c) Per‑syllable CHAR candidates (字).** For "fix syllable `p`" (req. #12) and the char portion of the unified stream: condition on the committed prefix (chars `<p`, reusing their cached KV), one forward, `top‑k` over `legal[p]` with true conditional `logP(char|prefix,input)`. This is `ranked[i]` (`slothd_e.py:241–247`) but joint‑conditioned and cache‑cheap (only the last step varies). Feeds `res.candidates` / `getLastWordCands` (`session.h:396–401`). The `decodeWithHints` interactive contract = teacher‑force the user's fix into the KV and re‑decode the tail only (AR makes `ChoosingCore.rescore` `core.h:340–418` exact rather than approximate).

**(d) WORD/PHRASE candidates (詞).** Local constrained beam width **b=6**, depth **K=2..4** from `p` over `legal[p..p+K-1]`, scored by cumulative masked joint log‑prob; keep continuations above `cut = max(0.06, 0.15·top)` (the exact cutoff from `phrases()` `slothd_e.py:281`). Depth‑1 gives chars, depth‑2/3/4 give words/phrases, **all on the same log‑prob scale** — subsumes both `phrases()` softmax‑product and `ensurePhrases()`/`phrasesScored` (`core.h:295–334`) with real AR joint probability.

**KV‑cache reuse** (what makes (a)–(d) affordable): prompt KV computed once and shared by all four; beams/phrase‑beams fork only their divergent suffix; char queries reuse the committed‑prefix KV and branch one step; an interactive re‑score truncates KV to the edited position and re‑decodes the tail — cheaper than the encoder's full re‑forward.

**Toneless / 音標皆可 (req. #10).** Two mechanisms, both already in‑tree, plus the AR win:
1. *Mask:* an unmarked syllable token selects the **tone‑union** legal set; a marked one uses the exact toned row; mixed marked/unmarked in one input just means per‑position different mask widths — falls out for free (`ㄋㄧㄏㄠㄇ→你好嗎`, `ㄨㄛㄕˋㄇㄠ→我是貓`).
2. *Training:* `gen_data.py:76` already emits toned (drop 0.0) / toneless (drop 1.0) / partial (drop U(0.3,0.8)) per sentence.
3. *AR joint prior closes the `呢嚆` collapse:* the encoder scored each toneless slot by an **independent marginal** over the whole union → `你好嗎` degraded to `呢嚆…`. The AR decoder scores each candidate **conditioned on the committed prefix**: `P(好|你) ≫ P(嚆|呢)`, so left context disambiguates the union. Target: match/beat warm‑start 84.9 toneless. *Residual:* space‑tone‑1 vs genuine‑toneless share a token; add an optional soft "space⇒tone‑1‑row preferred" logit bias (the spec's small tone‑1 prior) so toned accuracy doesn't regress.

### Outside the span (free)

When `p == N_syl` the mask lifts and the **same model, same loop** continues unconstrained — association, next‑word, continuation, and emoji are the *natural tail of the sequence*, not a separate subsystem (Design A's payoff, kept).

- **Next‑word / continuation / mid‑sentence prediction (req. #11):** short free beam (width 4, depth ≤4) from the current cache. Because the model just generated `這是樹懶`, its continuation is genuinely sentence‑conditioned — richer than `assoc.h` bigram counts. Mid‑sentence prediction = the same op invoked while composing, conditioned on committed tail + current best conversion.
- **Emoji (req. #6) — dual source (both designs converge):** (1) **model‑native** — emoji are native vocab tokens and `gen_data.py` preserves them verbatim in `out`, so fine‑tuning teaches `樹→🌲`, `書→📖` directly, arriving on the same log‑prob scale as words; (2) **`emoji_map.tsv` fallback** — a curated head‑word→emoji table (CLDR zh‑Hant annotations) injected at a calibrated log‑prob just below the model's own emoji mass, covering the tail the model hasn't seen and the arrow glyphs iOS shows (⬆️ for 上). Ship the map first (unblocked), let the model subsume it.
- **Personalization prior:** keep `assoc.h` + `assoc_tc.tsv` bigram engine and the calibrated learn store as **additive log‑prob bonuses** (per‑user habits the frozen LM can't hold), folded into ranking below — not replaced.

### Unified ranking (the labelless iOS stream, req. #5)

Every candidate is a token sequence with a sequence log‑prob. One scale, one ordered stream, no 字/詞/聯 group labels:

```
score(c) = (1/|c|^α)·Σ_k logP(tok_k | prefix, input)   # length-normalized joint LP (α≈0.6–0.7, tuned)
         + λ_len  · len_prior(kind)                     # small: guarantees the full sentence leads (#3)
         + λ_learn· learn_bonus(c)                      # calibrated learn store, slothd_e.py:139–173
         + λ_assoc· assoc_bonus(c)                      # personal bigram, assoc.h (outside-span)
         + λ_emoji· emoji_bias(c)                       # slightly negative → emoji rides just behind its anchor word
```

This merges Design A's additive‑prior list (learn+assoc+type) with Design B's length‑normalization intuition into one formula. α makes a 7‑char sentence comparable to a 1‑char candidate; `len_prior` gives the full‑sentence conversion a deliberate lead so it wins slot 0 (a **presentation pin** boxed/highlighted), with progressively shorter prefix‑truncations (`這是`/`這`) falling just behind on the same scale (req. #4 — remove the old hard cap of 3). **Dedup by surface string, keep max score across sources** (a char reachable as both sentence‑position‑0 and a char‑candidate collapses to one). **Emoji anchoring:** insert an emoji immediately after the first surviving candidate whose text is its anchor word (overrides pure score by ≤1 slot → `樹` then `🌲`). **Stability:** don't reshuffle already‑visible leading items on minor rescore; only the tail past the expand‑fold re‑sorts freely.

Exact order the bar consumes: **(1)** headline sentence (boxed); **(2)** n‑best sentence alternates + prefix fallbacks interleaved by score; **(3)** char + word/phrase + emoji for the focused syllable, merged by score, deduped; **(4)** outside‑span predictions/continuations/emoji when `p==N_syl` or mid‑sentence. Concatenate, stable‑sort by `score` → the single relevance‑ranked stream (`是 說 上 ⬆️ 聲 收 剩 樹`; after picking 樹: `樹 🌲 屬 書 📖 上午 食物`). The model, not the UI, decides char‑vs‑word‑vs‑emoji order.

### Code‑switch (req. #13) — resolved: keep the segmenter front door

- **Learned inline:** trained on `gen_data.py` pairs whose `out` copies literal English/numbers (`world cup`, `louisa`); during decode English is **force‑copied** from `⟨en⟩…⟨/en⟩`, so the model can never transliterate given English — it only *scores* the surrounding zh context, giving natural boundaries in the ranking.
- **Segmenter stays as the front door** (Design B, over Design A's "drop it"). The LM sees syllable tokens + English runs, never raw dachen keystrokes; `Segmenter::segment()` (`segment.h:83–209`) keeps doing cheapest‑cost DP + the `zhuyin-wins` reclassification (`:187–207`) to turn the keystream into that token stream. Dropping it (A) would force the LM to learn keymap/tone‑cost disambiguation it was never pretrained for — a bad trade, and it discards tuned infra encoding the `zhuyin-wins` memory. **v2 upgrade:** model‑scored segmentation (`DESIGN-E.md:133–151`) — the GPT rescores the top‑N DP lattice segmentations by prefill log‑prob and picks the winner; additive, gated to only‑when‑the‑segmenter‑reports‑near‑tie so common cases stay single‑pass (grafts A's ambition without its cost on every keystroke).

---

## Dataset design

Corpus root: `corpus_e3.txt` (~1.1M TW sentences on the workstation). All generation pairs flow through `gen_data.py` → JSONL → `gen_convert.py`; this section specifies the **weighting and the four extensions** needed for full iOS parity.

### 1. Frequency‑faithful TW weighting

Emit sentences at their **natural corpus frequency** (do not dedup) so common everyday phrases dominate — the web‑crawl corpus under‑represents them (the `重新→崇心` failure `common_words.py` was built to fix). Keep `common_words.py`'s upweighting: `jieba`‑segment the corpus, count 2–4‑char Han words, and inject each frequent word repeated **~linear in log‑frequency** (its current `min(max_repeat, 20 + c//8)` schedule) as clean in‑isolation word examples. This gives the model strong word‑level defaults on the same scale as sentence context.

### 2. CKIP word segmentation → word units + frequencies (new tooling)

The repo has only `jieba` (`common_words.py`); the task requires **CKIP‑grade TW segmentation**. Add `word_freq.py` using **`ckip-transformers`** (BERT‑based TW word segmenter, far better on Traditional Chinese named entities/compounds than jieba) to produce two artifacts from `corpus_e3.txt`:
- `word_freq.tsv` — `word<TAB>count`, the TW word‑frequency table. Feeds (a) a **word‑frequency prior** blended into `len_prior`/word scoring, and (b) the WORD/PHRASE training targets below. Keep jieba as a fast fallback (already wired).
- **Word‑boundary supervision** for phrase pairs: for each sentence, CKIP boundaries mark which output spans are single words, used to synthesize word/phrase‑only pairs (input = that word's syllables, output = the word) so `phrasesScored` has clean joint‑probability word units. Upweight by `word_freq`.

### 3. zh‑EN code‑switch preservation (already correct, verify coverage)

`gen_data.py` already preserves inline English/number runs literally in both `in` and `out` (`build_input` emits the ASCII‑alnum run verbatim; punctuation is skipped only in the INPUT). Extension: wrap those runs in `⟨en⟩…⟨/en⟩` in the serialized template so the decoder's force‑copy has explicit sentinels. **Augment** with the existing `synth_codeswitch.py` / `inject_codeswitch.py` generators to ensure enough `world cup`/`louisa`‑style mixed sentences that English copy‑through is robust (guard against catastrophic forgetting of English during zh‑heavy FT — Design A/B risk).

### 4. Emoji corpus + word↔emoji (new data)

- **Model‑native:** retain corpus sentences containing emoji (do not strip) so the LM learns emoji as high‑probability continuation tokens after their anchor words. If `corpus_e3.txt` is emoji‑sparse, mine an emoji‑bearing TW social corpus slice or synthesize `word 🌲`‑style continuation pairs from the map below.
- **`emoji_map.tsv`** (`anchor_word<TAB>emoji[ emoji…]`, e.g. `樹\t🌲`, `書\t📖 📚`, `貓\t🐱`, `聲\t🔊`, `上\t⬆️`) built from **CLDR / Unicode emoji‑annotation zh‑Hant** filtered to high‑frequency anchors (top few hundred, for latency). Used both as the runtime fallback (Decoding → outside) and to synthesize **word→emoji outside‑span training pairs**: input = anchor word's syllables, target = `word emoji` (emoji in the *outside* region), reinforcing emoji as free‑gen output.

### 5. Prediction / continuation pairs (outside‑span training — new)

To make outside‑span free generation strong (req. #11 mid‑sentence + post‑commit), add **truncated‑input continuation pairs**: for a corpus sentence `S` split at a random word boundary into prefix `S₁` / continuation `S₂`, emit an example whose **input span is only the syllables of `S₁`** and whose **target is `S₁ S₂`** — so the model learns to convert the typed prefix (inside) and *keep generating* the continuation (outside) past `p==N_syl`. Also emit pairs where the input is empty/1‑word and the target is a common next word (post‑commit prediction). Draw continuations frequency‑faithfully and bias toward `assoc_tc.tsv`/CKIP‑frequent bigrams so free‑gen agrees with the personalization prior.

### 6. Toneless / partial‑tone augmentation (already present, keep)

`gen_data.py:76` already yields three variants per sentence — toned / fully toneless / partial `U(0.3,0.8)`. This is the +3pt‑toneless lever. Keep it for **every** category above (conversion, word‑only, continuation, emoji) so toneless tolerance holds across the whole surface, not just full sentences. `augment_toneless.py` remains the reference for the dropout distribution.

### Exact training example FORMAT (input template → target, with loss masking)

Extend `gen_convert.py`'s `GenDS` to the new template. Prompt (input span, up to `⟨sep⟩`) is **loss‑masked** (`-100`); only the output span (post‑`⟨sep⟩` through `⟨eos⟩`) carries loss — exactly the current `lab = [-100]*len(p) + c` at `gen_convert.py:44`.

```
# Conversion (toned/toneless/partial), code-switch, emoji — one row per gen_data.py line:
input_ids : ⟨bos⟩ ⟨s:ㄓㄜ⟩ ⟨s:ㄕˋ⟩ ⟨s:ㄕㄨ⟩ ⟨s:ㄌㄢˇ⟩ ⟨en⟩world cup⟨/en⟩ ⟨sep⟩ 這 是 樹 懶 world cup ⟨eos⟩
labels    :  -100   -100    -100    -100     -100        -100 …    -100     -100  這 是 樹 懶 world cup ⟨eos⟩
#            └─────────────────────── prompt (masked) ───────────────────────────────┘ └──── supervised ────┘

# Word-only (CKIP unit, upweighted by word_freq):        in=⟨s:ㄕㄤˋ⟩⟨s:ㄨˇ⟩ ⟨sep⟩ 上午 ⟨eos⟩
# Continuation (outside-span free-gen): in=syl(S₁) ⟨sep⟩ S₁S₂ ⟨eos⟩   (loss on S₁S₂ — inside+outside both supervised)
# Word→emoji (outside):                 in=⟨s:ㄕㄨˋ⟩ ⟨sep⟩ 樹 🌲 ⟨eos⟩
```

Key format decisions vs current `gen_convert.py`: substitute syllable substrings with `⟨s:*⟩` ids at collate time (map via `syl_vocab.json`); replace `" →"` with `⟨sep⟩`; wrap English in `⟨en⟩…⟨/en⟩`; keep the length‑1 Han output filter; hard‑mask `⟨s:*⟩` ids from the label/output space so syllable tokens are never a target. Everything else (loss masking, `≤96` token cap, bf16, OneCycle) is reused.

---

## Training recipe

**Base & FT strategy.** Warm‑start SmolLM‑Chinese‑180M; **full fine‑tune** (180M is small and the syllable‑token embeddings + zh conversion behavior are a large distribution shift — LoRA under‑fits the new embedding rows). For the Qwen‑0.5B teacher, **LoRA + low‑LR full‑FT of the new embedding rows** to avoid catastrophic forgetting of its English/emoji strength (Design A/B risk #6). Resize embeddings by +1474 syllable rows + any missing Han single‑char rows; **warm‑start** each syllable row = mean of `tone_union_chars(syl)` pretrained char embeddings (`bert_convert.py:97–110` logic, imported as `from eval_arch import tone_union_chars`); untie or hard‑mask syllable rows from the output projection.

**Objectives.**
1. **Causal LM cross‑entropy on the output span only** (prompt masked) — the base conversion + code‑switch + emoji + continuation objective. This single objective, over the dataset above, already teaches inside‑conversion *and* outside‑free‑gen because the continuation/word→emoji pairs put targets on both sides of `⟨sep⟩`.
2. **Preserve/reinforce outside‑span free generation:** the continuation pairs (§5) are the explicit signal; additionally interleave a small fraction of **pure‑LM sentences** (no `⟨sep⟩`, plain corpus text) so the base's pretrained continuation ability doesn't erode under conversion‑heavy FT. Ratio ~5–10%.
3. **Distillation (student inherits the ceiling):** after the 0.5B teacher is trained, do **sequence‑level KD** (teacher‑forced 0.5B outputs as extra targets) **+ logit KD on the masked (constrained) distributions** into the 180M student. The masked‑distribution KD is what transfers homophone ranking quality; teacher‑forced sequences transfer fluency. Optionally distill 180M → ~60–70M ternary student (libslothe) for low‑end devices, keeping the encoder drafter.

**Hyperparameters (Stage‑0 accuracy, bf16).** LR 2e‑4 (student full‑FT) / 5e‑5 (teacher LoRA); AdamW wd 0.01; OneCycle pct_start 0.03; batch 32 (accumulate to ~256 effective on 2×5090); ≤96 tokens; 1–3 epochs over the full corpus (scale `--max-pairs` up from the current 800k to the full ~1.1M×3 variants). Decode presets: beam B=8, phrase b=6 K=3, char top‑k=8 (`kLastCands`), free‑beam depth ≤4 — all runtime‑tunable per device tier.

**Curriculum.** (1) toned‑only conversion to establish the syllable‑token embeddings and 1:1 alignment; (2) add toneless/partial variants (the harder disambiguation); (3) add word‑only + code‑switch + emoji + continuation; (4) personalization‑calibration pass to fit `α, λ_len, λ_learn, λ_assoc, λ_emoji`. Gate before advancing each stage.

**Eval (extend `eval_arch.py` / gates).** `eval_arch.py` already scores toned+toneless sentence accuracy with the tone‑union constraint on `eval/testset.tsv`; add an `--arch gpt` path (constrained AR decode) and new gate metrics:
- **Conversion accuracy toned / toneless** (existing; target ≥ warm‑start 86.8 / 84.9; **falsify early if the AR model can't beat the encoder here**).
- **Candidate‑ranking quality** — a small human‑rated candidate‑order set (there is no ground‑truth ranking); measure MRR of the intended char/word and whether the boxed sentence wins slot 0.
- **Prediction quality** — next‑word top‑k accuracy on held‑out continuations (outside‑span).
- **Code‑switch** — exact English copy‑through rate on mixed sentences.
- **Emoji recall** — anchor→emoji top‑k on the emoji eval slice.
- Reuse `gate_g2pw.py` / `gate_slothe_ternary.py` structure; parity re‑measured in `eval/ui-parity` after each phase (MEMORY: parity is measured, 23/27 baseline — don't guess).

---

## Deployment & latency

Accuracy first, latency later — each stage measured before the next.

- **Stage 0 — accuracy ceiling (bf16, no compromise).** Full‑precision constrained beam. Gate on the eval above. Ship nothing until Stage 0 ≥ warm‑start encoder (86.8/84.9) and clears 免選字 72–74. **If the AR model can't beat the encoder at bf16, the thesis is falsified cheaply** — kill early.
- **Stage 1 — quantize.** GGUF **Q4_K_M** (180M ≈ 110 MB); re‑gate, accept ≤1–2 pt regression. One GGUF over **ggml/llama.cpp** serves all frontends, converging the 4 frontends off ORT (`slothing-libslothe-deploy` goal). Ternary (libslothe TQ, ≈60–70 MB) for low‑end.
- **Stage 2 — encoder‑as‑drafter speculative decoding (the centerpiece, both designs).** The shipping **25M ternary SlothE encoder is the draft proposer**: its per‑position argmax (`slothd_e.py:245–256`) proposes the whole 1:1‑aligned sentence in **one non‑AR forward**; the GPT **verifies in a single batched forward**, accepting the longest matching masked prefix and resampling only from the first disagreement. Because the span is strictly 1:1 aligned and the encoder is already ~83–87% right, **acceptance is high → near‑encoder latency with GPT joint‑probability accuracy**. Reuses *all* encoder infra (the `decode()`/mask path is literally the drafter); no new model. Degrades gracefully to full GPT decode on hard toneless disagreements. Measure accept‑rate on the toneless slice specifically (the case AR is meant to fix, where drafter↔GPT most disagree); if low, train a **toneless‑tuned drafter**.
- **Stage 3 — distill** (only if still short): 180M → ~60–70M student on the GPT's constrained outputs; keep the encoder drafter.

**Per‑frontend:**
- **Web (WASM / WebGPU):** llama.cpp‑WASM / WebGPU over the single GGUF; KV‑cache + per‑beam fork is the riskiest surface in‑browser — the encoder‑drafter hybrid is the hedge (draft in the 25M ONNX/WASM path already shipping, verify with a handful of GPT forwards). Falls back to the current encoder if WebGPU is unavailable.
- **Android (ggml / onnx):** llama.cpp JNI over the GGUF; native `getUnifiedCands()` in `session.h` computes the scored unified list once in C++ so both mobile frontends stay thin.
- **Linux (fcitx5 / ibus, native):** same GGUF over ggml behind the `slothd`/`Decoder` seam; `slothd_e.py`'s socket protocol is unchanged.

---

## Migration & rollout

Phased from today's encoder to the GPT, across web‑touch, Android, fcitx5/ibus, and the desktop keyboard model — **the GPT drops in behind `decoder.h`'s `decode`/`decodeWithHints`/`phrasesScored`/`learn` with zero UI change**, so most iOS‑parity UI work ships on the *current* encoder in parallel.

- **Phase 0 — frontend parity on today's encoder (UNBLOCKED):** contextual space (一聲↔下一個, `moveHighlight` already exists), expand chevron, mount‑bar‑by‑model, remove prefix cap of 3, honest toneless hint. No model dependency.
- **Phase 1 — unified candidate bar (encoder log‑prob scoring):** merge sentence/prefix/字/詞/predict producers into one scored+deduped `UnifiedCand[]` using the encoder's per‑position log‑softmax as the interim single axis (α·lengthBonus hand‑tuned to offset the encoder's independent‑marginal over‑confidence). Ships the labelless stream *before* the GPT.
- **Phase 2 — emoji + prediction into the stream:** ship `emoji_map.tsv`, fold `AssocEngine` predictions + mid‑sentence trigger into the unified list. Data + FE only.
- **Phase 3 — toneless tone‑union in shared core:** segmenter accepts toneless bases (drop `toneOrSpace` requirement), decode masks against the tone‑union with a tone‑1 prior; web mirror. Encoder‑era fix; makes the `聲調可省略` hint truthful.
- **Phase 4 — GPT decoder swap (this doc):** train Stage 0–1, drop in behind `decoder.h`. It **replaces** per‑position argmax → joint constrained beam, single‑flip n‑best → real k‑best, `<en>` passthrough → real code‑switch + model‑native emoji, and inside‑masked / outside‑free = conversion / prediction. It **reuses** the phonetic mask + tone‑union, segmenter DP, corpus/g2pW alignment, learn store, and hint channel. `α·lengthBonus` shrinks toward 0 as scores become honest joint log‑probs.

**A/B against the warm‑start encoder:** run the encoder and the GPT behind the same `Decoder` seam, route by a per‑device flag, and compare in `eval/ui-parity` (add cases for unified ranking, toneless mixtures, emoji‑in‑stream, contextual space) plus the new candidate‑ranking/prediction/code‑switch/emoji gates. Speculative decoding makes this literally an A/B *within one path* — the encoder is the drafter, the GPT the verifier, so the encoder's answer is always available as the fallback and the accept‑rate is the live quality/latency telemetry. Keep the web adaptive‑model split (AUTO/桌面/行動) and Android‑always‑touch / Linux‑always‑keyboard; none of this touches the desktop keyboard model.

---

## Requirement coverage matrix

| iOS‑parity requirement | Mechanism in this design | Dataset support | Status |
|---|---|---|---|
| #1 Raw bopomofo field | Frontend paints raw syllables; model never eagerly converts into the field | — (UI) | Covered (frontend, deliverable #1) |
| #2 Bar on top of keyboard | Frontend mount by `uiMode()==='ios'`; model supplies `UnifiedCand[]` | — (UI) | Covered (frontend) |
| #3 Full‑sentence‑first | Constrained best sentence = beam[0], presentation‑pinned slot 0 (boxed) | Full‑sentence conversion pairs (freq‑faithful) | **Covered** |
| #4 Progressive shorter fallbacks | Beam prefix‑truncations + n‑best alternates on one scale; prefix cap removed | Same pairs; word‑only pairs | **Covered** |
| #5 Unified relevance‑ranked stream | Single length‑normalized log‑prob axis merges char/word/phrase/sentence/emoji/predict; dedup; no group labels | Conversion + CKIP word + emoji + continuation pairs | **Covered** (honest joint scores are the GPT payoff) |
| #6 Emoji in stream | Model‑native emoji tokens (outside‑span) + `emoji_map.tsv` fallback, anchored after their word | Emoji‑preserving corpus + word→emoji pairs + CLDR map | **Covered** |
| #7 Expand chevron | Frontend toggles scroll‑row↔grid over the same ordered list | — (UI) | Covered (frontend) |
| #8 Space = 一聲 / 注 watermark | Frontend label; model unaffected | — (UI) | Covered (frontend) |
| #9 Space = 下一個 when choosing | Frontend → `moveHighlight(+1)` over the unified stream (native primitive exists) | — (UI) | Covered (frontend) |
| #10 Toneless / 音標皆可 | Tone‑union mask for unmarked syllables + AR joint prior (fixes 呢嚆) + tone‑1 soft bias | toned/toneless/partial 3‑variant augmentation (`gen_data.py:76`) | **Covered** (target ≥84.9) |
| #11 Prediction (mid + post‑commit) | Outside‑span free generation = native next‑word/continuation; `AssocEngine` as additive prior | Truncated‑input continuation pairs + `assoc_tc.tsv` | **Covered** |
| #12 Fix a character | Per‑syllable masked top‑k conditioned on committed prefix (`decodeWithHints` = teacher‑force + re‑decode tail) | Conversion pairs (per‑position supervision) | **Covered** (model side; Android ↓window is frontend) |
| #13 zh‑EN code‑switch inline | `⟨en⟩…⟨/en⟩` force‑copy; segmenter front door; learned copy‑through | English‑preserving + synth code‑switch pairs | **Covered** |
| Char (字) candidates | Decode (c): masked top‑k joint conditional | Conversion pairs | **Covered** |
| Word/phrase (詞) candidates | Decode (d): local constrained beam depth K, joint log‑prob | CKIP word‑only pairs, `word_freq.tsv` prior | **Covered** |
| Backspace reclaim committed | Frontend + `session.h` reclaim surface (encoder‑era task) | — | Needs‑work (frontend/core, not model) |
| Symbol strip (符) + emoji category | `emoji_map.tsv` + symbol menu categories | CLDR emoji annotations | Covered (data + frontend) |
| Single‑model coverage | One AR LM owns conversion + candidates + emoji + prediction + code‑switch via inside/outside decode | Whole dataset | **Covered** (design thesis) |

---

## Open questions & risks

1. **Enforced 1:1 vs the LM's instinct to merge/split.** The hard pointer forbids emitting a 2‑char word for one syllable or skipping a slot. Ablate hard‑pointer vs soft alignment; quantify whether 1:1 hurts fluency or *is* the 免選字 legality guarantee. Han‑length‑1 output masking + guaranteed single‑char token coverage is the mitigation (Architecture).
2. **Tied‑embedding corruption / syllable tokens leaking to output.** If the base ties input/output embeddings, +1474 syllable rows also enter the output projection; must untie or hard‑mask them from every output step. Verify per base (SmolLM and Qwen differ).
3. **AR latency on long input.** A 40‑syllable sentence is 40 sequential steps × beam on a phone; the encoder does it in one forward. Stages 2–3 exist for this; if speculative accept‑rate is low on toneless (the very case AR fixes), latency reverts toward full AR — may need a toneless‑tuned drafter.
4. **Toneless union explosion in the beam.** Fully toneless 40‑syllable input gives huge per‑slot legal sets; the beam may thrash. Prune the union by a unigram prior before masking.
5. **Cross‑source log‑prob calibration.** `α, λ_len, λ_learn, λ_assoc, λ_emoji` must make the bar order match iOS intuition; no ground‑truth ranking exists → needs the human‑rated candidate‑order eval, not just top‑1 accuracy. One global `α`/`len_prior` may not simultaneously satisfy "sentence leads" and "short fallbacks visible" across sentence lengths → consider length‑bucketed priors or a tiny learned reranker.
6. **Catastrophic forgetting of English/emoji** during zh‑heavy FT — the very code‑switch/emoji advantages that motivate the base. Mitigate with mixed‑ratio data, the 5–10% pure‑LM interleave, LoRA/low‑LR on the teacher, and explicit copy‑through/emoji‑recall gates.
7. **CKIP dependency & throughput.** `ckip-transformers` is a BERT‑based segmenter — slower than jieba over 1.1M sentences; run it once offline to build `word_freq.tsv`, keep jieba as the runtime/fallback. Confirm it's installable in `~/slothc_venv`.
8. **`emoji_map.tsv` / model‑native emoji coverage.** Verify SmolLM's byte‑BPE actually emits the target emoji frequently enough to rank them in‑stream; if not, lean on the map fallback and add synthesized word→emoji pairs. Confirm no critical Han char falls to multi‑byte fragments (audit against `phonetic_table.tsv`).
9. **Distillation‑recovers‑quality bet.** The whole ship‑180M‑teach‑with‑0.5B plan assumes the student inherits enough of the teacher's edge over the encoder. If the 180M student erases the 0.5B's advantage, the flagship‑only 0.5B rung (over budget) becomes the only way to realize the ceiling — accept that as the escape hatch, not the default.
10. **Model‑scored segmentation cost (v2).** Enumerating serializations multiplies forwards on ambiguous runs; gate strictly to segmenter‑reported near‑ties so common cases stay single‑pass.

**Files this design touches / reuses:** `/home/luigi/sloth-zhuyin-linux/model/gen_data.py` (add `⟨en⟩` sentinels; else reused for toned/toneless/partial + code‑switch + emoji), `gen_convert.py` (extend `GenDS` template: `⟨sep⟩`, `⟨s:*⟩` ids, English wrap, keep loss‑mask + length‑1 Han filter; base default already `Mxode/SmolLM-Chinese-180M`), new `syl_vocab.json` (built from `phonetic_table.tsv`'s 1474 keys), `phonetic_table.tsv` + `engine/slothd/slothd_e.py:83–137` (mask/tone‑union/typo/phrases/learn reused verbatim), `bert_convert.py:97–110` (`tone_union_chars` warm‑start, applied to syllable rows), new `word_freq.py` (CKIP; `common_words.py` jieba kept as fallback), new `emoji_map.tsv` (CLDR zh‑Hant), `eval_arch.py`/`gate_g2pw.py`/`gate_slothe_ternary.py` (add `--arch gpt` + ranking/prediction/code‑switch/emoji gates), `engine/common/segment.h` (kept as front‑door tokenizer; DP arbitration retained, model‑scored lattice is v2), `engine/common/assoc.h` + `assoc_tc.tsv` + `slothd_e.py:139–173` learn store (folded into unified ranking), `engine/common/decoder.h` `Decoder` seam + `core.h ChoosingCore` (unchanged; GPT drops in behind). Ship path: one Q4_K_M/TQ GGUF over ggml for web (WASM/WebGPU), Android (JNI), and fcitx5/ibus, with the 25M ternary encoder retained as the speculative drafter.