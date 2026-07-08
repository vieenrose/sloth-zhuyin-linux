"""Slothing web IME backend.

Loads the 34M SlothLM GGUF and decodes typed bopomofo syllables to Traditional
Chinese under a phonetic-legality GBNF grammar -- the same prompt/grammar
slothingd uses, reimplemented on llama-cpp-python so the Space is self-contained
(no C++ daemon to build).
"""
import os

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from huggingface_hub import hf_hub_download
from llama_cpp import Llama, LlamaGrammar

REPO = "Luigi/slothlm-34m-zhuyin"
GGUF = "slothlm-34m-zhuyin-Q4_0.gguf"
TABLE = os.path.join(os.path.dirname(__file__), "phonetic_table.tsv")
TONES = "ˊˇˋ˙"

app = FastAPI()

_llm = None
_tonal = {}     # exact syllable  -> [chars]
_toneless = {}  # tone-stripped   -> [chars] (union of tones)


def _load_table():
    with open(TABLE, encoding="utf-8") as f:
        for line in f:
            syl, _, rest = line.rstrip("\n").partition("\t")
            if not rest:
                continue
            chars = list(rest)  # one Han char per codepoint
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
        path = hf_hub_download(REPO, GGUF)
        _llm = Llama(model_path=path, n_ctx=2048, verbose=False)
    return _llm


def _candidates(syl):
    """Phonetically-legal characters for a syllable; exact tone first, else the
    toneless union; an unknown token (English) is a literal passthrough."""
    if syl in _tonal:
        return _tonal[syl]
    base = "".join(c for c in syl if c not in TONES)
    if base in _toneless:
        return _toneless[base]
    return [syl]  # code-switch: keep English verbatim


def _gbnf(positions):
    esc = lambda s: s.replace("\\", "\\\\").replace('"', '\\"')
    lines = ["root ::=" + "".join(f" p{i}" for i in range(len(positions)))]
    for i, cands in enumerate(positions):
        alts = " | ".join(f'"{esc(c)}"' for c in cands)
        lines.append(f"p{i} ::= {alts}")
    return "\n".join(lines)


class DecodeReq(BaseModel):
    syllables: list[str]
    context: str | None = None


@app.on_event("startup")
def _startup():
    _load_table()
    _model()  # warm the model so the first request isn't slow


@app.post("/api/decode")
def decode(req: DecodeReq):
    syllables = [s for s in req.syllables if s]
    if not syllables:
        return {"sentence": "", "positions": []}
    positions = [_candidates(s) for s in syllables]

    user = " ".join(syllables)
    if req.context:
        user = req.context + "＿＿＿\n" + user
    prompt = (
        "<|im_start|>system\n注音轉繁體中文。<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    grammar = LlamaGrammar.from_string(_gbnf(positions), verbose=False)
    out = _model()(prompt, grammar=grammar, max_tokens=128, temperature=0.0,
                   stop=["<|im_end|>"])
    sentence = out["choices"][0]["text"].strip()
    return {"sentence": sentence, "positions": positions}


app.mount("/static", StaticFiles(directory=os.path.join(
    os.path.dirname(__file__), "static")), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(os.path.dirname(__file__),
                                     "static", "index.html"))
