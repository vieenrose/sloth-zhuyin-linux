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

## Results

| Model | Size (Q4_K_M) | Test 1 output | Test 2 output | Accuracy | Notes |
|---|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | 469MB | 你**號邪斜**你 (1/5) | 我在吃飯的時候 ✅ (7/7) | 8/12 (67%) | Weakest overall; sometimes fails formatting entirely without grammar constraint |
| Qwen2.5-1.5B-Instruct | 1.1GB | **擬**好謝謝你 (4/5) | 我**再**吃飯的時候 (6/7) | 10/12 (83%) | |
| LFM2.5-1.2B-Instruct | 698MB | 你好**邪邪**你 (3/5) | 我在吃飯的時候 ✅ (7/7) | 10/12 (83%) | Fastest per-token generation (~34ms/tok) |
| gemma-3-1b-it | 769MB | 你好謝**斜**你 (4/5) | 我**再**吃飯的時候 (6/7) | 10/12 (83%) | Slowest per-token generation (~58ms/tok) despite being smallest param count tested here |
| MiniCPM4-0.5B | 266MB | 你好**邪邪**妳 (2/5) | 我在**池**飯的**十**候 (5/7) | 7/12 (58%) | Worst overall; also needs ~3x the token budget per character vs. other models' tokenizers, so a naive `max_tokens` truncates output mid-grammar (finish_reason=length) unless sized generously |

## Takeaways

- None of these models are reliably correct without task-specific
  fine-tuning or richer context — confirms the GeneInput/Chinese-GPT
  papers cited in `RESEARCH.md`: generic prompting of an off-the-shelf
  instruct model caps out around 60-85% char accuracy on this kind of
  probe.
- Grammar-constrained decoding (GBNF restricting output to real candidate
  characters) is a hard requirement regardless of model choice — it
  guarantees well-formed output and turns "generate Chinese text" into
  "classify each position," which is a much easier task for small models.
  It also surfaces per-model quirks like MiniCPM4's higher tokens-per-char
  ratio, which would otherwise silently truncate output in production.
- **Decision: LFM2.5-1.2B** is the best size/speed/quality tradeoff of the
  five tested — ties the top accuracy (83%, alongside Qwen2.5-1.5B and
  gemma-3-1b-it) while being smaller than 1.5B and generating fastest
  per-token of the three 83%-accuracy models. This is what `slothingd`
  (Milestone 2) will ship with.
