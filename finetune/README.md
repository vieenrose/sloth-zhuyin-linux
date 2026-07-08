# Fine-tuning pipeline (v0.3)

Prepares a domain-specialised LFM2.5-230M for the conversion task. Runs on a
**GPU workstation**, not the CPU dev box. Motivation: off-the-shelf LLMs align
phonetic and text representations poorly until explicitly trained on
conversion (Huawei PY-GEC, cosine 0.26 → 0.82); the baseline eval shows the
stock model already helps (+16pt sentence acc) but introduces its own misses
(音樂→陰月, 個→个) that targeted training should fix.

## 0. Corpus

Need a plain-text file of Traditional-Chinese sentences (2–18 chars),
one per line — ideally Taiwan-register (TW news, PTT, Wikipedia-zh-TW,
subtitle corpora). Not vendored (size/licensing); bring your own.

## 1. Generate the dataset (on the dev box — needs libchewing + eval/harvest)

```sh
gcc eval/harvest.c -o eval/harvest $(pkg-config --cflags --libs chewing)
python3 -m venv llm/venv && llm/venv/bin/pip install pypinyin
llm/venv/bin/python finetune/gen_dataset.py \
    --sentences corpus.txt --out finetune/train.jsonl
```

Each sentence yields up to three chat examples in the daemon's exact prompt
format so training matches serving:
- **SELECT** (`系統:選字。`) — pick the sentence from real per-position
  libchewing candidate lists (the grammar-constrained task).
- **Z2T** (`注音轉繁體中文。`) — zhuyin → sentence, direct phonetic mapping.
- **T2Z** (`繁體中文轉注音。`) — the reverse alignment task.

Sentences in `eval/testset.tsv` are auto-excluded (no train/test leakage), as
are ones whose chewing harvest doesn't reconstruct the sentence.

## 2. Train (on the workstation)

```sh
pip install "transformers>=4.45" "trl>=0.11" peft datasets accelerate bitsandbytes
python finetune/train_lora.py --data finetune/train.jsonl \
    --out out/lfm2-230m-slothing
```

LoRA (r=16) on attn+MLP projections of `LiquidAI/LFM2.5-230M` (the base, not
the -GGUF). ~230M params + light LoRA fits comfortably on a single 16–24GB
GPU; a few epochs over ~100k examples is well under an hour on an A10G/L4.

## 3. Merge + convert to GGUF

```sh
python -c "from peft import AutoPeftModelForCausalLM as M; \
m=M.from_pretrained('out/lfm2-230m-slothing').merge_and_unload(); \
m.save_pretrained('out/merged'); \
from transformers import AutoTokenizer as T; \
T.from_pretrained('out/lfm2-230m-slothing').save_pretrained('out/merged')"

python llm/llama.cpp/convert_hf_to_gguf.py out/merged \
    --outfile out/lfm2-230m-slothing-f16.gguf --outtype f16
llm/llama.cpp/build/bin/llama-quantize \
    out/lfm2-230m-slothing-f16.gguf \
    llm/models/lfm2.5-230m-slothing/LFM2.5-230M-slothing-Q4_0.gguf Q4_0
```

## 4. A/B against the baseline

```sh
# baseline daemon already benchmarked in eval/README.md; now the tuned one:
./engine/slothingd/build/slothingd \
    -m llm/models/lfm2.5-230m-slothing/LFM2.5-230M-slothing-Q4_0.gguf &
python3 eval/run_eval.py            # compare sentence/char acc + latency
```

Ship the tuned GGUF (point `run-slothingd.sh` at it) only if it beats the
83% / 95.8% baseline without a latency regression. Record the new numbers in
`eval/README.md` and `MODEL_BENCHMARKS.md`.

## License note

LFM2.5 weights are under LiquidAI's "LFM Open License" — confirm its
fine-tune/redistribution terms before publishing a tuned GGUF. Training
scripts and dataset generator here are ours (repo license).
