"""Render assets/demo.gif from the real answers captured by scripts/capture_demo.py.

Only the typing and waiting are animated; the answers and sources are the real output.
Run: uv run --with pillow python scripts/render_demo.py   (pillow is already a dependency)
"""

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
W, H, PAD, SIZE = 1000, 520, 18, 17
BG, FG, DIM, GREEN, RED, BLUE, YELLOW = "#0d1117", "#e6edf3", "#7d8590", "#3fb950", "#f85149", "#58a6ff", "#d29922"
font = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", SIZE)
LINE = SIZE + 6
data = json.loads((ROOT / "assets" / "demo_capture.json").read_text(encoding="utf-8"))

frames, durations = [], []
lines: list[list[tuple[str, str]]] = []  # each line is a list of (text, colour) pieces


def snap(ms: int) -> None:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 30], fill="#161b22")
    for i, c in enumerate(("#f85149", "#d29922", "#3fb950")):
        d.ellipse([14 + i * 22, 9, 26 + i * 22, 21], fill=c)
    d.text((W // 2 - 90, 6), "gated-rag demo", font=font, fill=DIM)
    visible = lines[-((H - 50) // LINE):]
    for row, pieces in enumerate(visible):
        x = PAD
        for text, colour in pieces:
            d.text((x, 42 + row * LINE), text, font=font, fill=colour)
            x += d.textlength(text, font=font)
    frames.append(img)
    durations.append(ms)


def clip(text: str, n: int = 104) -> str:
    return text if len(text) <= n else text[: n - 3] + "..."


snap(700)
lines.append([("# Same question, two roles. Access is enforced before ranking.", DIM)])
snap(1800)
for item in data:
    prompt = f"[{item['role']}] Ask: "
    lines.append([])
    for i in range(0, len(item["q"]) + 1, 2):
        lines[-1] = [(prompt, BLUE), (item["q"][:i], FG)]
        snap(45)
    snap(500)
    lines.append([("  searching, re-ranking, answering ...  (time-lapse, takes ~30-60 s on a CPU)", YELLOW)])
    snap(1500)
    lines.pop()
    answer_colour = RED if item["a"].lower().startswith("i don't know") else GREEN
    first = item["a"].split(". ")[0].rstrip(".") + "."
    lines.append([("  " + clip(first), answer_colour)])
    snap(1400)
    if answer_colour == GREEN:
        for src in item["sources"].splitlines()[:2]:
            lines.append([(clip(src.replace("education_examination_regulations", "oer").replace(" > Education and Examination Regulations (OER) 2025-2026", ""), 108), DIM)])
        snap(2600)
    else:
        lines.append([("  (nothing the 'public' role may see answers this: the internal document was never retrieved)", DIM)])
        snap(2600)
    lines.append([])
lines.append([("Local models, no cloud. github.com/AlgoCodology/gated-rag", GREEN)])
snap(3500)

out = ROOT / "assets" / "demo.gif"
frames[0].save(out, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)
print(out, len(frames), "frames", round(out.stat().st_size / 1024), "KB")
