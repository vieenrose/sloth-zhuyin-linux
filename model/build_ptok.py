#!/usr/bin/env python3
"""Word-piece tokenizer for the Apple-style predictor: BPE over the TW corpus
(common 2-4 char words -> single tokens => one forward predicts a whole word) +
100 curated emojis (Apple ships 100). ~16k vocab."""
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
EMOJIS = list("😀😁😂🤣😊😍🥰😘😉😎🤔😏😌😔😢😭😤😠😡🥺😴🤤😷🤒🤕🤢🤮🥵🥶😳🤯😱😨😰😥"
              "👍👎👌✌️🙏👏🙌💪🤝🖐️❤️🧡💛💚💙💜🖤🤍💔💯🔥✨⭐🌟💫🎉🎊🎁🎂🌸🌺🌻🌷🌹"
              "🍎🍊🍋🍉🍓🍒🍑🥭🍍🥝☕🍵🍺🍻🍷🍔🍕🍜🍚🍙🚗🚕🚙🏠🏢📱💻⌚📷🎮⚽🏀🎵🎶")
def main():
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tr = trainers.BpeTrainer(vocab_size=16000, min_frequency=3,
                             special_tokens=["<pad>","<bos>","<eos>","<unk>"] + EMOJIS,
                             initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train(["corpus_e3.txt"], tr)
    tok.save("predictor_tok.json")
    v = tok.get_vocab_size()
    print(f"vocab={v}  emojis added={len(set(EMOJIS))}")
    for s in ["今天天氣","我想喝咖啡","謝謝你","大家好😊"]:
        ids = tok.encode(s).ids
        print(f"  {s!r} -> {len(ids)} tokens: {[tok.id_to_token(i) for i in ids]}")
if __name__ == "__main__": main()
