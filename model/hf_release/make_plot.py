#!/usr/bin/env python3
"""Score-vs-latency scatter for the model card. Self-contained SVG, no deps."""

# honest held-out (500 fresh zh-TW sentences) + BOOX (SD662) per-6-syllable decode ms
# latency: 4M/12M measured (ORT int8); 25M ternary projected from measured-shape
#          bitnet TQ2_0/I2_S kernel benchmarks (marked with *).
MODELS = [
    # name,           params, latency_ms, proj, mianxuan, homophone, toneless, color
    ("4M int8",       "4M",   9.06, False, 70, 83, 74, "#6b7280"),
    ("12M int8",      "12M",  13.3, False, 72, 82, 79, "#2563eb"),
    ("25M ternary",   "25M",  9.0,  True,  76, 86, 77, "#dc2626"),
]

W, H = 680, 440
L, R, T, B = 74, 28, 46, 66          # margins
PX0, PX1 = L, W - R
PY0, PY1 = T, H - B
XMIN, XMAX = 8.0, 14.5
YMIN, YMAX = 66.0, 80.0

def x(v): return PX0 + (v - XMIN) / (XMAX - XMIN) * (PX1 - PX0)
def y(v): return PY1 - (v - YMIN) / (YMAX - YMIN) * (PY1 - PY0)

s = []
s.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'viewBox="0 0 {W} {H}" font-family="-apple-system,Segoe UI,Roboto,sans-serif">')
# always-light background card so it reads on both HF themes
s.append(f'<rect x="0" y="0" width="{W}" height="{H}" rx="10" fill="#ffffff"/>')
s.append(f'<text x="{L}" y="26" font-size="16" font-weight="700" fill="#111827">'
         f'Held-out quality vs. on-device latency (BOOX SD662)</text>')

# gridlines + y ticks (免選字 %)
for yv in range(66, 81, 2):
    yy = y(yv)
    s.append(f'<line x1="{PX0}" y1="{yy:.1f}" x2="{PX1}" y2="{yy:.1f}" stroke="#e5e7eb"/>')
    s.append(f'<text x="{PX0-10}" y="{yy+4:.1f}" font-size="12" fill="#6b7280" '
             f'text-anchor="end">{yv}</text>')
# x ticks (ms)
for xv in range(8, 15, 1):
    xx = x(xv)
    s.append(f'<line x1="{xx:.1f}" y1="{PY0}" x2="{xx:.1f}" y2="{PY1}" stroke="#f3f4f6"/>')
    s.append(f'<text x="{xx:.1f}" y="{PY1+20}" font-size="12" fill="#6b7280" '
             f'text-anchor="middle">{xv}</text>')
# axis labels
s.append(f'<text x="{(PX0+PX1)/2:.0f}" y="{H-14}" font-size="13" fill="#374151" '
         f'text-anchor="middle">latency — ms / 6-syllable decode  (lower = faster)</text>')
s.append(f'<text x="18" y="{(PY0+PY1)/2:.0f}" font-size="13" fill="#374151" '
         f'text-anchor="middle" transform="rotate(-90 18 {(PY0+PY1)/2:.0f})">'
         f'免選字 (whole-sentence) %  (higher = better)</text>')
# axis frame
s.append(f'<rect x="{PX0}" y="{PY0}" width="{PX1-PX0}" height="{PY1-PY0}" '
         f'fill="none" stroke="#9ca3af"/>')

# Pareto arrow: 4M -> 25M ternary (up, same latency), the headline story
s.append(f'<line x1="{x(9.06):.1f}" y1="{y(70)-10:.1f}" x2="{x(9.0):.1f}" '
         f'y2="{y(76)+12:.1f}" stroke="#dc2626" stroke-width="1.5" '
         f'stroke-dasharray="4 3" opacity="0.6"/>')

# points
for name, tag, lat, proj, mx, ho, tl, col in MODELS:
    cx, cy = x(lat), y(mx)
    s.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="8" fill="{col}" '
             f'stroke="#ffffff" stroke-width="2"/>')
    star = "*" if proj else ""
    # two-line label fully above the point (name higher, stats just above point),
    # except 4M which sits low so its label goes below the point.
    if name == "4M int8":
        name_y, sub_y = cy + 24, cy + 39
    else:
        name_y, sub_y = cy - 30, cy - 14
    s.append(f'<text x="{cx:.1f}" y="{name_y:.1f}" font-size="13" font-weight="700" '
             f'fill="{col}" text-anchor="middle">{name}{star}</text>')
    s.append(f'<text x="{cx:.1f}" y="{sub_y:.1f}" font-size="11" fill="#6b7280" '
             f'text-anchor="middle">免{mx} · 同{ho} · 無{tl}</text>')

# projection note under the title (avoids the crowded bottom axis area)
s.append(f'<text x="{L}" y="42" font-size="11" fill="#9ca3af">'
         f'* 25M ternary latency projected from measured-shape TQ2_0 / I2_S kernels</text>')
s.append('</svg>')
open("score_vs_latency.svg", "w").write("\n".join(s))
print("wrote score_vs_latency.svg")
