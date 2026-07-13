#!/usr/bin/env python3
"""LoRA SFT for LFM2.5-230M on the Sloth IME conversion dataset.

Run on a GPU workstation (not this CPU box). Expects a JSONL of chat
examples from gen_dataset.py. Produces a LoRA adapter; merge + GGUF-convert
per finetune/README.md, then A/B on eval/run_eval.py.

  pip install "transformers>=4.45" "trl>=0.11" peft datasets accelerate bitsandbytes
  python finetune/train_lora.py --data train.jsonl --out out/lfm2-230m-sloth
"""
import argparse

from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

BASE = "LiquidAI/LFM2.5-230M"  # base (not the -GGUF); fine-tune then convert


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="out/lfm2-230m-sloth")
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype="bfloat16")

    ds = load_dataset("json", data_files=args.data, split="train")

    # Small model -> a light LoRA on the attention/MLP projections is plenty.
    peft_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    cfg = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=2,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=20,
        save_strategy="epoch",
        bf16=True,
        packing=True,
        max_length=1024,
        # TRL applies the model's chat template to the "messages" field.
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model, args=cfg, train_dataset=ds, peft_config=peft_cfg,
        processing_class=tok,
    )
    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print(f"LoRA adapter saved to {args.out}")


if __name__ == "__main__":
    main()
