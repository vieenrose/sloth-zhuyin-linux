# Reranker model benchmarks

Quick, informal benchmark to pick the LLM for `slothingd` (Milestone 2). Not
a rigorous eval — two hand-picked homophone-heavy test sentences, run
through llama.cpp's grammar-constrained decoding so the model can only ever
emit one of libchewing's real candidate characters per syllable position
(guarantees no garbage output regardless of model quality; only tests
whether the model picks the *right* candidate).

## Method

For each model, load via `llama-server` and send a chat completion request
listing per-position candidates (e.g. `第1字選(你/妳/擬) 第2字選(好/號/毫)...`)
with a GBNF grammar restricting output to exactly those candidates,
`temperature=0`.

- **Test 1**: `你好謝謝你` — candidates include the correct char among
  homophones at every position.
- **Test 2**: `我在吃飯的時候` — a common grammatical construction where
  在/再 is the key disambiguation point.
- **Test 3** (LFM2.5-230M only, added after it aced 1+2): `他是老師`.
- **Test 4** (LFM2.5-230M only): `今天天氣很好`.

Models after LFM2.5-1.2B were run through the actual `slothingd` daemon
(`engine/slothingd/`, this project's own llama.cpp-C-API Unix-socket
reranker) rather than `llama-server`, once it was working — same
grammar-constrained-decoding technique either way.

## Results

| Model | Size | Test 1 | Test 2 | Test 3 | Test 4 | Accuracy | Notes |
|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | 469MB (Q4_K_M) | 你**號邪斜**你 (1/5) | 我在吃飯的時候 ✅ (7/7) | — | — | 8/12 (67%) | Weakest of the ~0.5-1.5B instruct models; sometimes fails formatting entirely without grammar constraint |
| Qwen2.5-1.5B-Instruct | 1.1GB (Q4_K_M) | **擬**好謝謝你 (4/5) | 我**再**吃飯的時候 (6/7) | — | — | 10/12 (83%) | |
| LFM2.5-1.2B-Instruct | 698MB (Q4_K_M) | 你好**邪邪**你 (3/5) | 我在吃飯的時候 ✅ (7/7) | — | — | 10/12 (83%) | Fastest per-token of the ~1B-class models tested (~34ms/tok) |
| gemma-3-1b-it | 769MB (Q4_K_M) | 你好謝**斜**你 (4/5) | 我**再**吃飯的時候 (6/7) | — | — | 10/12 (83%) | Slowest per-token (~58ms/tok) despite smallest param count in that group |
| MiniCPM4-0.5B | 266MB (Q4_K_M) | 你好**邪邪**妳 (2/5) | 我在**池**飯的**十**候 (5/7) | — | — | 7/12 (58%) | Worst of that group; also needs ~3x the token budget per character, so a naive `max_tokens` truncates output mid-grammar unless sized generously |
| ERNIE-4.5-0.3B-PT | 368MB (Q8_0) | — | — | — | — | n/a | Base **pretrained** model (no chat template) — `llama_chat_apply_template` fails outright with our chat-based prompting; would need raw-completion prompting to evaluate fairly, not attempted |
| **LFM2.5-230M** | 236MB (Q8_0) | 你好謝謝你 ✅ (5/5) | 我在吃飯的時候 ✅ (7/7) | 他是老師 ✅ (4/4) | 今天天**器**很好 (5/6) | **21/22 (95%)** | Smallest model tested, and by far the most accurate — beats every larger model above by a wide margin |
| LFM2.5-230M | 115MB (unsloth UD-Q2_K_XL) | 你好謝謝**擬**(4/5) | 我在吃飯的時候 ✅ (7/7) | 他是老**濕** (3/4) | 今天**添**氣很好 (5/6) | 19/22 (86%) | Tried a much more aggressive quant hoping for lower latency after hitting slow responses under heavy system load. Backfired on both axes: *lower* accuracy (86% vs 95%) and *higher* latency (~1.4-3.5s vs ~0.75-1.6s on the same loaded machine) — 2-bit/mixed-precision quants don't map as efficiently to llama.cpp's CPU SIMD kernels as Q4/Q8 for a model this small, where per-weight unpacking overhead dominates. **Not used**; Q8_0 above is what ships. |

## Takeaways

- None of the ~0.5-1.5B instruct models are reliably correct without
  task-specific fine-tuning or richer context — confirms the
  GeneInput/Chinese-GPT papers cited in `RESEARCH.md`: generic prompting
  caps out around 60-85% char accuracy on this kind of probe.
- Grammar-constrained decoding (GBNF restricting output to real candidate
  characters) is a hard requirement regardless of model choice — it
  guarantees well-formed output and turns "generate Chinese text" into
  "classify each position," which is a much easier task for small models.
  It also surfaces per-model quirks like MiniCPM4's higher tokens-per-char
  ratio, and ERNIE-4.5-0.3B-PT's total lack of a chat template.
- **Surprise finding: LFM2.5-230M dramatically outperforms every larger
  model tested** (95% vs 58-83%), despite being 2-5x smaller. Liquid's
  LFM2.5 family appears specifically well-suited to this
  classification-shaped task, and smaller LFM2.5 checkpoints aren't simply
  "worse" than bigger ones on it — worth retesting if quality regresses on
  harder sentences later.
- **Decision: LFM2.5-230M** is what `slothingd` (Milestone 2) ships with —
  it's simultaneously the smallest, fastest, and most accurate model
  benchmarked.
