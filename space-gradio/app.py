"""Slothing web IME -- Gradio SDK build.

Same decode logic as the Docker/FastAPI build (space/app.py): SlothLM GGUF via
llama-cpp-python, phonetic-legality GBNF grammar from phonetic_table.tsv. The
UI is built from plain Gradio components (button grid + radio groups) instead
of a custom HTML/JS frontend, so it runs on HF's managed Gradio runtime rather
than a scheduled Docker container -- trading a little UI polish for a much
more reliable path to actually being live.
"""
import os
import threading

import gradio as gr

TONES = "ˊˇˋ˙"
TABLE = os.path.join(os.path.dirname(__file__), "phonetic_table.tsv")
REPO = "Luigi/slothlm-34m-zhuyin"
GGUF = "slothlm-34m-zhuyin-Q4_0.gguf"

# Dàqiān layout key -> (symbol, slot): 0 initial, 1 medial, 2 final.
DACHEN_ROWS = [
    [("1","ㄅ",0),("q","ㄆ",0),("a","ㄇ",0),("z","ㄈ",0),("2","ㄉ",0),("w","ㄊ",0),
     ("s","ㄋ",0),("x","ㄌ",0),("e","ㄍ",0),("d","ㄎ",0),("c","ㄏ",0)],
    [("r","ㄐ",0),("f","ㄑ",0),("v","ㄒ",0),("5","ㄓ",0),("t","ㄔ",0),("g","ㄕ",0),
     ("b","ㄖ",0),("y","ㄗ",0),("h","ㄘ",0),("n","ㄙ",0)],
    [("u","ㄧ",1),("j","ㄨ",1),("m","ㄩ",1)],
    [("8","ㄚ",2),("i","ㄛ",2),("k","ㄜ",2),(",","ㄝ",2),("9","ㄞ",2),("o","ㄟ",2),
     ("l","ㄠ",2),(".","ㄡ",2),("0","ㄢ",2),("p","ㄣ",2),(";","ㄤ",2),
     ("/","ㄥ",2),("-","ㄦ",2)],
]
TONE_BUTTONS = [("ˊ","6"), ("ˇ","3"), ("ˋ","4"), ("˙","7"), ("一聲","")]

_tonal, _toneless = {}, {}
_llm, _llm_lock = None, threading.Lock()


def _load_table():
    with open(TABLE, encoding="utf-8") as f:
        for line in f:
            syl, _, rest = line.rstrip("\n").partition("\t")
            if not rest:
                continue
            chars = list(rest)
            _tonal[syl] = chars
            base = "".join(c for c in syl if c not in TONES)
            seen = set(_toneless.get(base, []))
            merged = _toneless.setdefault(base, [])
            for c in chars:
                if c not in seen:
                    seen.add(c)
                    merged.append(c)


def _model():
    global _llm
    if _llm is None:
        with _llm_lock:
            if _llm is None:
                from huggingface_hub import hf_hub_download
                from llama_cpp import Llama
                path = hf_hub_download(REPO, GGUF)
                _llm = Llama(model_path=path, n_ctx=2048, verbose=False)
    return _llm


def _candidates(syl):
    if syl in _tonal:
        return _tonal[syl]
    base = "".join(c for c in syl if c not in TONES)
    return _toneless.get(base, [syl])


def _gbnf(positions):
    esc = lambda s: s.replace("\\", "\\\\").replace('"', '\\"')
    lines = ["root ::=" + "".join(f" p{i}" for i in range(len(positions)))]
    for i, cands in enumerate(positions):
        lines.append(f"p{i} ::= " + " | ".join(f'"{esc(c)}"' for c in cands))
    return "\n".join(lines)


def decode_syllables(syllables):
    from llama_cpp import LlamaGrammar
    positions = [_candidates(s) for s in syllables]
    user = " ".join(syllables)
    prompt = (
        "<|im_start|>system\n注音轉繁體中文。<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
    )
    grammar = LlamaGrammar.from_string(_gbnf(positions), verbose=False)
    out = _model()(prompt, grammar=grammar, max_tokens=128, temperature=0.0,
                   stop=["<|im_end|>"])
    return out["choices"][0]["text"].strip(), positions


# ---------------------------------------------------------------------------
# Gradio callbacks. State: cur = [init, med, final] (pending syllable),
# committed = list of syllable strings typed so far.

def key_press(cur, committed, sym, slot):
    cur = list(cur)
    cur[slot] = sym
    return cur, committed, "".join(committed) + "".join(cur)


def tone_press(cur, committed, tone):
    if not any(cur):
        return cur, committed, "".join(committed)
    committed = committed + ["".join(cur) + tone]
    cur = ["", "", ""]
    return cur, committed, "".join(committed)


def backspace(cur, committed):
    cur = list(cur)
    if cur[2]:
        cur[2] = ""
    elif cur[1]:
        cur[1] = ""
    elif cur[0]:
        cur[0] = ""
    elif committed:
        committed = committed[:-1]
    return cur, committed, "".join(committed) + "".join(cur)


def clear_all(cur, committed):
    return ["", "", ""], [], ""


def do_convert(cur, committed, toneless):
    syllables = list(committed)
    if any(cur):
        syllables = syllables + ["".join(cur)]
    if not syllables:
        return ("", [], gr.update(visible=False), gr.update(visible=False),
                gr.update(visible=False), gr.update(visible=False),
                gr.update(visible=False))
    if toneless:
        syllables = ["".join(c for c in s if c not in TONES) for s in syllables]
    sentence, positions = decode_syllables(syllables)
    sel = [ (positions[i].index(ch) if ch in positions[i] else 0)
           for i, ch in enumerate(sentence) ] if len(sentence) == len(positions) else [0]*len(positions)
    updates = []
    for i in range(5):  # up to 5 visible segment pickers
        if i < len(positions) and len(positions[i]) > 1:
            updates.append(gr.update(choices=positions[i], value=positions[i][sel[i]],
                                     label=f"第{i+1}字", visible=True))
        else:
            updates.append(gr.update(visible=False))
    return sentence, positions, *updates


def recompose(*seg_values):
    return "".join(v for v in seg_values if v)


def commit_to_output(output, composed):
    return (output or "") + (composed or ""), "", ["", "", ""], [], ""


THEME = gr.themes.Soft(primary_hue="orange", neutral_hue="stone")

with gr.Blocks(theme=THEME, title="Slothing — Web Zhuyin IME") as demo:
    gr.Markdown(
        "# 🦥 Slothing — 注音輸入法\n"
        "由 **34M** 從零訓練的模型（[SlothLM](https://huggingface.co/Luigi/slothlm-34m-zhuyin)）"
        "解碼，每個字都受「音對得上」的文法約束，不會亂猜字。\n\n"
        "[GitHub](https://github.com/vieenrose/sloth-zhuyin-linux) · "
        "libchewing-free · CPU only"
    )

    output = gr.Textbox(label="輸出", lines=3, interactive=False)
    preedit = gr.Textbox(label="正在輸入（注音）", interactive=False)

    cur_state = gr.State(["", "", ""])
    committed_state = gr.State([])
    positions_state = gr.State([])

    with gr.Row():
        toneless_cb = gr.Checkbox(label="免聲調", value=False)
        convert_btn = gr.Button("轉換 🡒", variant="primary")
        clear_btn = gr.Button("清除")
        back_btn = gr.Button("⌫")

    gr.Markdown("### 注音鍵盤")
    key_buttons = []
    for row in DACHEN_ROWS:
        with gr.Row():
            for key, sym, slot in row:
                b = gr.Button(sym, min_width=40)
                key_buttons.append((b, sym, slot))
    with gr.Row():
        tone_buttons = [gr.Button(label or "1", min_width=40) for label, _ in TONE_BUTTONS]

    for b, sym, slot in key_buttons:
        b.click(key_press, [cur_state, committed_state, gr.State(sym), gr.State(slot)],
               [cur_state, committed_state, preedit])
    for b, (label, tone) in zip(tone_buttons, TONE_BUTTONS):
        b.click(tone_press, [cur_state, committed_state, gr.State(tone)],
               [cur_state, committed_state, preedit])
    back_btn.click(backspace, [cur_state, committed_state],
                   [cur_state, committed_state, preedit])
    clear_btn.click(clear_all, [cur_state, committed_state],
                    [cur_state, committed_state, preedit])

    gr.Markdown("### 選字（點擊修改每個字）")
    seg_radios = [gr.Radio(choices=[], label=f"第{i+1}字", visible=False)
                 for i in range(5)]
    composed_box = gr.Textbox(label="組成的句子", interactive=False)
    commit_btn = gr.Button("確認加入輸出", variant="primary")

    convert_btn.click(do_convert, [cur_state, committed_state, toneless_cb],
                      [composed_box, positions_state, *seg_radios])
    for r in seg_radios:
        r.change(recompose, seg_radios, composed_box)
    commit_btn.click(commit_to_output, [output, composed_box],
                     [output, composed_box, cur_state, committed_state, preedit])

    demo.load(lambda: _load_table() or None, None, None)
    demo.load(lambda: threading.Thread(target=_model, daemon=True).start(), None, None)

if __name__ == "__main__":
    demo.launch()
