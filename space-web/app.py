"""Slothing web IME -- custom UI hosted on the Gradio runtime.

The Docker Space path kept failing at HF hardware scheduling ("unable to
schedule"). Gradio SDK Spaces schedule reliably, so this serves our real
custom IME frontend (ui.html: on-screen + physical keyboard, click-to-swap
segment editing) inside a gr.Blocks, bridging to the SlothLM decode through
three hidden components (in / button / out). Same decode as the Docker build:
llama-cpp-python + phonetic-legality GBNF grammar.
"""
import json
import os
import threading

import gradio as gr

TONES = "ˊˇˋ˙"
HERE = os.path.dirname(__file__)
TABLE = os.path.join(HERE, "phonetic_table.tsv")
UI = open(os.path.join(HERE, "ui.html"), encoding="utf-8").read()
REPO = "Luigi/slothlm-34m-zhuyin"
GGUF = "slothlm-34m-zhuyin-Q4_0.gguf"

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


def decode_bridge(payload):
    """payload: JSON {"syllables":[...]} -> JSON {"sentence":..,"positions":..}"""
    try:
        req = json.loads(payload) if payload else {}
    except Exception:
        return json.dumps({"sentence": "", "positions": []})
    syllables = [s for s in req.get("syllables", []) if s]
    if not syllables:
        return json.dumps({"sentence": "", "positions": []})
    from llama_cpp import LlamaGrammar
    positions = [_candidates(s) for s in syllables]
    user = " ".join(syllables)
    prompt = ("<|im_start|>system\n注音轉繁體中文。<|im_end|>\n"
              f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n")
    grammar = LlamaGrammar.from_string(_gbnf(positions), verbose=False)
    out = _model()(prompt, grammar=grammar, max_tokens=128, temperature=0.0,
                   stop=["<|im_end|>"])
    return json.dumps({"sentence": out["choices"][0]["text"].strip(),
                       "positions": positions})


with gr.Blocks(title="Slothing — Web Zhuyin IME",
               css="footer{display:none!important}") as demo:
    gr.HTML(UI)
    bridge_in = gr.Textbox(visible=False, elem_id="sl-bridge-in")
    bridge_out = gr.Textbox(visible=False, elem_id="sl-bridge-out")
    bridge_btn = gr.Button(visible=False, elem_id="sl-bridge-btn")
    bridge_btn.click(decode_bridge, bridge_in, bridge_out)

# Load the phonetic table and warm the model at import time -- NOT on demo.load
# -- so a decode works regardless of how the app is invoked (browser session,
# API client, etc.). demo.load only fires on a browser page-load, which left
# the table empty for headless/API calls (every syllable then fell through to
# literal passthrough and the model just echoed the bopomofo back).
_load_table()
threading.Thread(target=_model, daemon=True).start()

if __name__ == "__main__":
    demo.launch()
