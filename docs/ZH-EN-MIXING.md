# 中英混排 — zh-TW / English mixed writing & typing conventions

*A convention study for the Sloth IME Zhuyin IME (2026-07-12). Authoritative ruleset behind the mixed-script segmenter/display behavior, derived from a multi-agent research pass over W3C CLReq, 教育部《重訂標點符號手冊》, pangu.js / 中文文案排版指北, and real zh-TW corpora, then adversarially reviewed.*

It answers two load-bearing questions and fixes two shipped bugs: `7-11`→`7兒11` (hyphen read as zhuyin ㄦ) and `Expected: 你好`→`Expected：你好` (halfwidth colon widened after an English word).


## The verdicts (what the IME does)

### Space between Chinese and English/numbers?

NO — the IME must NOT auto-insert a space at a Chinese↔English/number boundary. Default = faithful transcription: a space appears only if the user actually typed one. This is where the six facets genuinely diverge (see conflicts_resolved), and the verdict resolves it decisively for a keystream IME: (1) the authoritative ¼-em gap in CLReq is a TYPOGRAPHIC micro-space delivered by the layout engine / CSS text-autospace, not a literal ASCII 0x20, and an IME can only emit the wrong thing (a full ½-em breakable space); (2) CLReq co-editor Bobby Tung strongly opposes literal manual spaces (semantic pollution: T恤→T 恤, B計畫→B 計畫, dirty data, bad line breaks); (3) every native zh-TW 注音 IME (微軟新注音, 新酷音/libchewing, 自然, RIME) emits verbatim and none auto-space — pangu.js exists precisely because IMEs don't; (4) majority casual reality (LINE/PTT/Dcard/Threads) omits the space anyway; (5) auto-spacing corrupts 7-11, COVID-19, URLs, code, times. The competing pangu / 中文文案排版指北 web-copy school (a real, popular convention) belongs in an OPT-IN whole-buffer post-processor run on finished text, off by default — never inline during typing. NOTE: the shipped display.h currently auto-spaces (lines 82-85/116-117); that behavior should be moved behind the opt-in flag.

### Punctuation after an English word

HALFWIDTH ASCII, hugging the English token, with Western spacing (space AFTER, never before): 'Expected: 你好' not 'Expected：你好'; 'ratio 16:9'; '3:30'; 'v2.0'. Width follows the run the mark grammatically binds to — a colon/comma/period bound to an English label, identifier, time, or number stays literal ASCII. THE ONE OVERRIDE: sentence-terminal . ! ? that closes an overall-Chinese sentence become fullwidth 。！？ even when the last token is English ('我用的是 Python。'), because the mark belongs to the Chinese sentence, not the English word. So bind width to the mark's owning run, with a sentence-terminator exception.

### Punctuation in Chinese text

FULLWIDTH CJK punctuation ，。！？：；、「」『』（）——…, occupying one Han em with built-in side-bearing, hugging its neighbors with ZERO surrounding spaces. This is the default in zhuyin context and holds even when an English word abuts the mark (我買了 iPhone，很貴。 keeps a fullwidth comma). zh-TW uses Taiwan corner brackets 「」 (outer) / 『』 (nested) and centered glyphs — not zh-CN ""/''. Sci-tech house style MAY unify the period to ． or . (CLReq exception); ship 。 as default, ． as an option.

### Separators inside IDs / numbers / URLs

HALFWIDTH LITERAL, atomic. A maximal run of ASCII letters/digits joined by internal separators (- : / . ~ % $ , ' @ +) is ONE token emitted byte-for-byte, half-width, never split, never width-promoted, and never fed to the zhuyin parser. 7-11, 3:30, 2026/07/12, 02-1234-5678, COVID-19, PM2.5, v1.2.3, $19.99, 50%, NT$100, https://a.com/x, user@mail.com all pass through verbatim. The hyphen in 7-11 is NEVER zhuyin ㄦ; the colon in a time is NEVER fullwidth ：. Ranges are the one soft spot: preserve whatever half-width form the user typed (5-10 / 5~10), do NOT auto-convert to the formal fullwidth ～/— and never to ㄦ.

### IME default behavior

Mode-less, segmenter-driven: default punctuation to FULLWIDTH in a zhuyin/Han run and auto-emit HALFWIDTH inside an English/number run, decided by the DP segmenter's run classification — no manual 全形/半形 mode switch needed to get correct output. English passthrough (Shift / Caps Lock / Shift+Space overrides) defaults to 半形/halfwidth. No auto-spacing, no auto-capitalization, no language-ratio guessing, no retroactive re-widening of already-committed marks. Behavior stays deterministic and predictable — matching what 新注音/酷音 users expect — with width bound at commit time from the current run plus the sentence-terminal override.


## Canonical rules

| id | category | rule | output convention | conf |
|---|---|---|---|---|

| `SPACE-no-autospace` | spacing | The IME MUST NOT automatically insert a space at a Chinese↔English or Chinese↔number boundary. Emit exactly the characters keyed; a space appears only if the user typed one, and user-typed spaces (including those inside a product name like 'iPhone 15 Pro') are preserved verbatim. | Faithful transcription: Han run and adjacent Latin/number run concatenate with no injected separator. Default keystream behavior; the current auto-spacing in display.h must move behind an opt-in flag. | high |
| `SPACE-gap-is-typographic` | spacing | Recognize that the authoritative Han↔Western/numeral gap is ~¼ Han-em, applied by the typesetter/browser, adjustable, and suppressed at line start/end — a RENDERING space, not a content character. Any desired optical spacing is the consumer's job (CSS text-autospace / font metrics), not the IME's. | No codepoint inserted into the text stream to represent the gap; a literal ASCII 0x20 is the WRONG representation (it is ~½-em and breakable). | high |
| `SPACE-pangu-optin-postproc` | spacing | If a product ever offers 盤古之白 spacing, it MUST be a separately-toggled post-processing pass over completed text (like pangu.js on a whole paragraph), never an inline auto-insert during typing. Even under pangu style: no space before ° or %, no space that splits an alphanumeric token, and no space around fullwidth punctuation. | Opt-in, off by default, whole-buffer. Number↔spelled-unit MAY be spaced (10 GB, 5000 元) but ° % stay glued and glued lexemes (T恤, B計畫) are never split. | high |
| `SPACE-none-around-fullwidth-punct` | spacing | Never place a space before or after fullwidth CJK punctuation ，。！？、：；「」（）——…. They carry their own internal side-bearing; an external space double-spaces them. Universal across formal, pangu, and casual conventions. | Fullwidth marks hug neighbors with zero surrounding spaces; any Han↔Latin gap rule stops at the punctuation and never fires across it. | high |
| `PUNCT-width-by-owning-run` | punctuation | Master rule: emit punctuation in the width of the language of the run/clause it grammatically belongs to — Chinese clause → fullwidth; embedded English word/label/identifier or number run → halfwidth. Decide from the OWNING run, not merely the single adjacent glyph. | Fullwidth set ，。：；！？「」『』（）——…、 for Chinese; halfwidth set , . : ; ! ? " ' ( ) - / for English/number runs. Bind width at commit time from the active run. | high |
| `PUNCT-halfwidth-after-english` | punctuation | Punctuation bound to an English word/label/identifier/number stays halfwidth ASCII, hugging that token, with Western spacing (space after, never before). THIS FIXES the fullwidth-colon bug. A colon after a Chinese word is fullwidth ： with no surrounding space. | After English token: ':' hugs the word, then Western space, then next token. After Chinese token: '：' with no spaces. Applies to , : ; ! ? bound to English clauses. | high |
| `PUNCT-sentence-terminal-override` | punctuation | A period/exclamation/question that TERMINATES an overall-Chinese sentence is fullwidth 。！？ even when the immediately preceding token is English. A terminal mark that closes a self-contained English sentence/quotation stays halfwidth. This is the one place the simple 'preceding-token' heuristic breaks — clause/sentence role wins for terminators. | Sentence-terminal . ! ? default to fullwidth 。！？ unless the entire terminated unit is English or a quoted-English clause. Sci-tech may unify to Western . (CLReq option). | medium |
| `PUNCT-taiwan-corner-quotes` | punctuation | In Chinese context use Taiwan corner brackets 「」 (outer) / 『』 (nested), including when quoting an English phrase inside a Chinese sentence. Use Western "" / '' only inside self-contained English text. Do not adopt zh-CN's opposite ""/'' nesting. | Fullwidth 「」『』 for quotations in Chinese context (one em, no extra spacing); halfwidth straight/curly quotes only within an English run. Vertical text rotates to ﹁﹂/﹃﹄. | high |
| `PUNCT-no-retroactive-rewrite` | punctuation | Bind width at commit time from the CURRENT run only. Do not retroactively re-widen an already-committed halfwidth mark because Chinese appears later, do not auto-capitalize, and do not perform paragraph-language-ratio guessing. Keep behavior deterministic and modal-equivalent, matching native 注音 IMEs. | The only context-sensitivity is run-type width binding + the sentence-terminal override; everything else is verbatim passthrough. | medium |
| `NUM-halfwidth-digits` | numbers-ids | Always emit Arabic numerals as half-width (0-9), never full-width (０-９), in modern zh-TW mixed text. Full-width digits read as an error today. | ASCII digits U+0030-0039; full-width digits never produced by the IME. | high |
| `NUM-literal-token-atomic` | numbers-ids | A maximal run of ASCII letters/digits joined only by internal separators (- : / . ~ % $ , ' @ +) is ONE literal English/number token: emit byte-for-byte half-width, never split it, never convert any internal separator to zhuyin or to fullwidth. This is the master rule that fixes the 7-11 bug and covers times (3:30), dates (2026/07/12), decimals/versions (3.14, v1.2.3), phones (02-1234-5678), symbols (50%, NT$100), acronyms (COVID-19, PM2.5, USB-C, C++), URLs and emails. | Segmenter locks the run as PASSTHROUGH; display emits it verbatim half-width. Internal separators live INSIDE the token so they can never surface standalone as zhuyin ㄦ or as a fullwidth mark. | high |
| `NUM-range-separator` | numbers-ids | Numeric/temporal ranges: preserve whatever half-width form the user typed (5-10 or 5~10); do NOT auto-convert to the formal fullwidth 甲式—(U+2014) / 乙式～(U+FF5E), and NEVER read the separator as zhuyin ㄦ. The load-bearing guarantee is: a range separator is never zhuyin. Distinguish from a brand hyphen (7-11), which is also preserved literally. | Casual default = keep typed half-width -/~. Formal fullwidth ～/— is an optional print style, not the digital default. Do not 'rangeify' a brand name. | medium |
| `NUM-letter-han-lexemes-glued` | numbers-ids | Taiwan-native single-letter+Han lexemes are written with NO space between the letter/digit and the Han character: K書, Q彈, T恤, 3C, A咖, PO文, AA制, call機. The letter stays half-width but glued. | Letter/digit directly glued to Han, no pangu space, no fullwidth. Worth a small IME lexeme dictionary. Note this is a lexical exception, distinct from a foreign word merely adjacent to Han (which under an opt-in pangu pass would take a space). | medium |
| `IME-dual-key-run-decides` | ime-behavior | For a key that is BOTH a zhuyin symbol and ASCII punctuation ('-'=ㄦ etc.), the run type decides: inside a number/English run the ASCII meaning wins (half-width literal); inside a valid zhuyin syllable the zhuyin meaning wins. A tone digit (3/4/6/7) does NOT count as English context, so genuine finals like ㄦ (這兒, 女兒) survive. This is the 7-11-bug fix at the segmentation layer. | DP segmenter offers the flanked-punct-as-literal edge cheaply (so it merges into the English run) only when a non-tone alnum neighbor sits on at least one side; a real zhuyin syllable still wins when one parses. Already implemented in segment.h lines 128-148. | high |
| `IME-segmenter-driven-width` | ime-behavior | Default punctuation to FULLWIDTH in a zhuyin/Han run and auto-emit HALFWIDTH inside an English/number run, driven by the segmenter's run classification — no manual 全形/半形 mode switch required for correct output. A manual toggle is offered only as an override. | Chinese context → ，。「」；：！？ (fullwidth default); English/number run → ,.-/:; (halfwidth), per PUNCT-halfwidth-after-english + sentence-terminal override. English/passthrough mode → always half-width. | medium |
| `IME-english-passthrough-affordances` | ime-behavior | Match the modal affordances zh-TW users already know: Shift = temporary/toggle English, Caps Lock = sustained English/英數, Shift+Space = 全形/半形 toggle. English/passthrough output defaults to HALF-WIDTH (半形); only when the user explicitly chooses 全形 do letters/digits widen. | Passthrough English = half-width lowercase-preserving ASCII; 半形 is the default width for letters/digits/space/punctuation. Sloth IME's mode-less design MAY make Shift mean 'force this run English' as long as the OUTPUT (half-width English) matches expectation. | high |


## How Sloth IME implements it


The behavior lives in the shared, frontend-free layer (`engine/common/segment.h` +
`engine/common/display.h`, mirrored in `space-static/segment.js` + `ime.js`), so the
fcitx5, IBus, and web frontends all agree.

**1. No auto-spacing at CJK↔Latin boundaries (`SPACE-*`).** The renderer, not the
IME, owns the ¼-em inter-script gap (CLReq / CSS `text-autospace`); an IME can only
emit a literal ASCII space, which CLReq's editor warns against (它會污染語意：`T恤`→`T 恤`)
and which corrupts `7-11`, URLs, and code. So the display layer now transcribes
verbatim — `我要去7-11採買`, not `我要去 7-11 採買`. (Previously `display.h` injected a
space around every ASCII run; that is removed. Pangu-style spacing, if ever wanted,
belongs in an opt-in whole-buffer post-processor, never inline.)

**2. A punctuation/number run is one atomic literal token (`NUM-literal-token-atomic`).**
A maximal run of ASCII letters/digits joined by internal separators (`- : / . ~ % $ , ' @ +`)
is a single English token emitted byte-for-byte: `7-11`, `3:30`, `2026/07/12`,
`COVID-19`, `user@mail.com`, `$19.99`, `50%`, `https://a.com/x`. The segmenter
achieves this with a *flanked-literal* rule: a punctuation key wedged next to an
English-context character (an alphanumeric that is **not** a tone key `3/4/6/7`)
is offered as a cheap literal so it merges into the run. The `-` key is therefore
literal in `7-11` but still zhuyin ㄦ in `這兒` (`5k4-`) / `女兒` (`sm3-6`), where
its neighbours are tone digits — and a genuine syllable always out-scores the
literal (`b.4`=ㄖㄡˋ) so real zhuyin is never lost.

**3. Punctuation width binds to the CLAUSE, not the previous token
(`PUNCT-halfwidth-after-english` + `PUNCT-width-by-owning-run`).** A mark is
halfwidth only inside a *pure-English clause* (the attached token is English **and**
no Chinese precedes it back to the last sentence terminator); otherwise it is the
fullwidth 微軟/酷音 mark. This keeps `Expected: 你好` halfwidth, keeps `我推薦 Python，因為…`
a **fullwidth** comma (it owns the Chinese clause), and makes `我用的是 Python。` a
fullwidth 。 — all without the risky sentence-terminal lookahead the review flagged
as the top misfire. `\` stays literal in English context (paths/regex/LaTeX), not `、`.

**Deliberately NOT done** (per the adversarial review): no auto-capitalization, no
language-ratio guessing, no retroactive re-widening of committed marks, and no
"halfwidth after *any* English run" (that would wrongly give `我推薦 Python，` a
halfwidth comma on a high-visibility glyph 注音 users expect to be ，).


## Where conventions genuinely diverge

- SPACING (the load-bearing disagreement): three facets (spacing, ime-behavior, corpus-evidence) say NO inline auto-space; two (punctuation's pangu-spacing rule, dataset-model's SPACE-HAN-LATIN-POLICY) default to inserting one. Resolved for a KEYSTREAM IME in favor of no-auto-space: the insert-space convention is a typesetting/post-processing rule (pangu.js runs on finished text with whole-token context), whereas an inline IME sees a partial keystream and would mis-tokenize and corrupt 7-11/URLs/code; CLReq's real gap is a ¼-em typographic micro-space an IME cannot emit; native zh-TW 注音 IMEs never auto-space; casual majority omits it. Pangu spacing is preserved as an explicit opt-in whole-buffer post-processor. (The shipped display.h currently auto-spaces — this is the divergent behavior to gate behind the flag.)
- SENTENCE-FINAL PERIOD: the simple 'width follows preceding token' heuristic (halfwidth after English) conflicts with '我用的是 Python。' taking a fullwidth 。. Resolved: run-language decides width EVERYWHERE except sentence-terminal . ! ?, where the mark belongs to the Chinese sentence/clause and goes fullwidth unless the whole terminated unit is English/quoted-English. Terminators bind to clause role, non-terminators bind to the adjacent run.
- RANGE SEPARATOR: 教育部 手冊 prescribes fullwidth 甲式—/乙式～ for ranges; real digital usage types half-width -/~. Resolved: preserve whatever half-width form the user typed (do not auto-convert, do not zhuyin-ify); fullwidth is an optional formal/print style, not the digital default.
- QUOTE NESTING: zh-TW 「」 outer / 『』 inner vs zh-CN opposite ""/''. Resolved: default to Taiwan corner brackets per MOE 2008; Western quotes only inside self-contained English.
- LETTER+HAN LEXEMES vs PANGU: pangu would space K 書 / 3C 產品; native writers glue lexemes (K書, 3C, T恤, AA制). Resolved: glued for the lexeme, with an optional-pangu space only at the true word boundary (before 產品), and only under the opt-in beautifier.
- PERIOD GLYPH: CLReq allows sci-tech to unify to Western '.' or fullwidth '．' to avoid 。/0 confusion; general text uses 。. Resolved: 。 default, ． / . as an option.


## Open questions

- No hard corpus counts were measured for space-vs-no-space frequency in PTT/Dcard; the 'casual majority omits the space' claim rests on the documented opt-in origin of the habit (1980s PE2 line-break workaround) plus the absence of any standard mandating a literal ASCII space. A quick corpus tally would upgrade SPACE-no-autospace's supporting evidence from inference to measurement.
- Per-app spacing policy: should the pangu opt-in be global, per-app (e.g. on in a Markdown/blog editor, off in a terminal/code field), or per-language-of-target-field? Native IMEs offer no such affordance; this is a product decision.
- Android port (Route B): phone IMEs (Gboard) DO auto-space/auto-capitalize in English mode, so BOOX/mobile users may expect a minimal English-run autospace that desktop users do not. Whether to diverge the default per-platform is unresolved; keep opt-in for now.
- Exact per-layout key→zhuyin collision tables beyond 大千 (倚天/Eten26, 許氏, 神通) need enumeration before the flanked-literal segmenter rule is fully layout-parameterized.
- Sentence-terminal detection is heuristic: deciding whether a trailing '.' closes a Chinese sentence vs is part of an English abbreviation/URL/version at the very end of input is ambiguous without look-ahead the IME may not have; may need the decoder's help or a conservative default.


## Dataset

`eval/zh-en-mixing.tsv` — 151 `input → expected` cases (spacing / punctuation / numbers-ids / ime-behavior) distilled from this study, including the two motivating bugs. Use as segmenter/display regression cases and as a checklist for mixed-script training-corpus augmentation.
