# Research: LLM-enhanced input methods

Notes gathered before starting the LLM reranker milestone (see plan at
`.claude/plans/hidden-zooming-tower.md` for the fcitx5/libchewing milestone
this follows).

## Baseline: what classic engines already do

libpinyin/libchewing/sunpinyin use an HMM/n-gram statistical language model
over a phonetic-to-character lattice — Viterbi search for the max-likelihood
character sequence given the whole pinyin/zhuyin string. Fast (sub-ms), but
short context window: struggles with rare words, abbreviations, and
long-range disambiguation. ([Libpinyin — Fcitx wiki](https://fcitx-im.org/wiki/Libpinyin))

## Where LLMs get inserted, per the literature

1. **Full-sentence P2C rescoring** — [Exploring and Adapting Chinese GPT to
   Pinyin Input Method (arXiv 2203.00249)](https://arxiv.org/pdf/2203.00249):
   a fine-tuned Chinese GPT beats n-gram baselines on "perfect" pinyin but
   degrades on abbreviated pinyin unless given pinyin-aware context
   concatenation and targeted fine-tuning — prompting a generic LLM as-is is
   not enough.
2. **Unified generative reframing** — [GeneInput (ACL Findings
   2024)](https://aclanthology.org/2024.findings-acl.218.pdf): treats
   key-sequence→character as one generative task, adds RLHF from user
   accept/reject signal instead of manual annotation, beats GPT-4 with a
   much smaller specialized model. Lesson: a small model fine-tuned for the
   IME task beats a big general model just prompted for it.
3. **On-device personalization via memory** — [HUOZIIME (arXiv
   2604.14159)](https://arxiv.org/pdf/2604.14159): a small on-device LLM
   with a hierarchical memory of the user's input history layered on top of
   the classic pinyin engine — directly relevant if we revisit
   personalization/vocabulary learning later.
4. **Latency budget is the hard constraint** — keyboard-decoder literature
   (KeySense, MobileLLM-Flash) targets **~100ms** input-to-candidate-update.
   A 0.5–1.5B quantized model doing single-pass rescoring on CPU fits that
   budget only if called selectively (sentence commit / ambiguous cases),
   not per-keystroke.

**Implication for our design**: keep libchewing's lattice/candidates as the
hot path; run the LLM only on the full uncommitted syllable buffer at
natural break points (space/punctuation/explicit convert key), rescoring
whole-sentence candidate paths rather than individual characters.
Personalization, if pursued, is a separate memory-augmented step, not baked
into the base reranker.

## Competitor check: 華碩智慧輸入法 (ASUS Smart Input)

- Free zhuyin IME, **Windows 10/11 only** (opened to non-ASUS PCs since Oct
  2023), via Microsoft Store. ([ASUS](https://www.asus.com/tw/content/smartinput/))
- Headline feature: automatic zh/en language detection so users never
  press Shift to switch modes when typing mixed Chinese/English.
- Real-world complaint ([review](https://blog.user.today/asus-smartinput-intro-bopomofo-pinyin/)):
  the auto-detection misfires on code/programming text, treating English
  code as Chinese-mode input and surfacing irrelevant word suggestions.
- Closed-source Windows-only tool — nothing to reuse directly, but it
  validates seamless zh/en code-switching as an unsolved user pain point.
  An LLM-based reranker, which has real context awareness, is a plausible
  way to do this better than ASUS's heuristic — potential stretch goal
  after the core sentence-prediction milestone.
