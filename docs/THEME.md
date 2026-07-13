# 樹懶 UI 主題 — 樹懶與樹(Sloth & Tree)

One visual identity across web / desktop / Android: **the sloth carries the
warmth, the tree carries the calm.** Bark browns and sloth-fur ambers are the
*primary* (interactive) colors; canopy greens are the *secondary* accent
(labels, focus, dividers). Dark mode is a **night forest**, not gray.

**Style principle: clean and minimalist.** The theme is carried by palette,
type, and the single 懶 mark — never by decoration. No patterns, no
illustrations in the UI, no gradients beyond the soft paper wash, at most one
green accent per view. If an element can be removed, remove it.

## Tokens

| Token | 樹梢 Light (default) | 夜森林 Dark | Role |
|---|---|---|---|
| `bg` | `#f6f1e8` warm paper | `#18211b` forest floor | page/panel bg |
| `panel` | `#fffdf9` | `#212c24` | cards, IME panel |
| `ink` | `#382d24` bark ink | `#e8e6d9` moonlight | main text |
| `muted` | `#97846f` | `#9aa78f` lichen | hints, labels |
| `bark` | `#8b5a2b` | `#c08347` amber | PRIMARY: buttons, selection, links |
| `bark2` | `#a86f38` | `#d29656` | primary hover |
| `leaf` | `#4f7942` fern | `#7fae6f` moss | SECONDARY: 聯 label, focus ring, checkbox, dividers |
| `border` | `#e2d3bc` | `#3a4a3c` moss edge | hairlines |
| `border2` | `#cdb694` branch | `#4a5d43` | strong borders |

Rationale: bark/amber survives 16-level grayscale and matches the 懶 tray
icon; leaf green is reserved for *secondary* signals so the shipped demo GIFs
(primary-color interactions) stay representative.

## Per-platform mapping

- **Web** (`space-static/index.html`): CSS custom properties; light by
  default, dark via `prefers-color-scheme` **and** a `?theme=dark|light` URL
  override (testing/screenshots). `color-scheme: light dark` is declared.
- **Desktop / fcitx5** (`packaging/themes/`): 樹懶·樹梢 (`Sloth IME`) and
  樹懶·夜森林 (`Sloth IME-Dark`) — panel bg/border/highlight from the token
  table. IBus draws with the system (KDE/GTK) theme and cannot be themed
  per-engine; the sloth identity there is the 懶 icon.
- **Android** (`values/colors.xml` + `values-night/`): sloth light/dark
  palettes follow the system day/night mode on LCD phones. **E-ink exception:
  ONYX devices keep the pure black-on-white palette** — flat fills, no hues —
  because 16-level grayscale + full-refresh ghosting punish anything subtler
  (see docs/SENIOR-KEYBOARD.md §1.5 for the sourced e-ink constraints).

## Motifs

- The 懶 glyph in Noto Serif TC, bark-colored, is the mark (web h1, tray icon,
  launcher icon).
- Dashed "branch" divider between preedit and strip uses `border2` (light) /
  `leaf`-tinted (dark).
- No gradients on e-ink; soft radial paper gradient allowed elsewhere.
