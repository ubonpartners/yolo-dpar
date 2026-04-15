#!/usr/bin/env python3
"""
Build a 3x2 side-by-side ReID comparison grid image from existing retrieval panels.

Output:
    images/reid_comparison_grid_3x2.png
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def render() -> Path:
    repo_root = Path(__file__).resolve().parent.parent
    images_dir = repo_root / "images"

    model_specs = [
        ("yolo26l.pt", "YOLO26L Stock (YOLO feats baseline)"),
        ("yolo26l-v10-240226.pt", "YOLO26L + Attributes (DPA)"),
        ("yolo26l-e2e-v10r-080426.pt", "YOLO26L + Attributes + ReID (DPAR)"),
    ]
    queries = ["q6508", "q7856"]

    panels: dict[tuple[str, str], Image.Image] = {}
    for q in queries:
        for model_file, _title in model_specs:
            name = f"ubonsyntheticloader_{model_file}_std-qg-v1_grid_2x1_{q}.jpg"
            path = images_dir / name
            if not path.exists():
                raise FileNotFoundError(f"Missing input image: {path}")
            panels[(q, model_file)] = Image.open(path).convert("RGB")

    first = panels[(queries[0], model_specs[0][0])]
    cell_w, cell_h = first.size

    pad = 18
    col_title_h = 52
    row_label_w = 84

    canvas_w = row_label_w + (3 * cell_w) + (4 * pad)
    canvas_h = col_title_h + (2 * cell_h) + (3 * pad)
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    title_font = _font(22)
    row_font = _font(24)

    # Column titles.
    for col_idx, (_model_file, title) in enumerate(model_specs):
        x0 = row_label_w + pad + col_idx * (cell_w + pad)
        title_w, title_h = _text_size(draw, title, title_font)
        tx = x0 + max(0, (cell_w - title_w) // 2)
        ty = pad + max(0, (col_title_h - title_h) // 2)
        draw.text((tx, ty), title, fill=(18, 24, 38), font=title_font)

    # Rows + images.
    for row_idx, q in enumerate(queries):
        y0 = col_title_h + pad + row_idx * (cell_h + pad)

        # Row label (query id)
        label = q.upper()
        lw, lh = _text_size(draw, label, row_font)
        lx = max(8, (row_label_w - lw) // 2)
        ly = y0 + max(0, (cell_h - lh) // 2)
        draw.text((lx, ly), label, fill=(55, 65, 81), font=row_font)

        for col_idx, (model_file, _title) in enumerate(model_specs):
            x0 = row_label_w + pad + col_idx * (cell_w + pad)
            panel = panels[(q, model_file)]
            if panel.size != (cell_w, cell_h):
                panel = panel.resize((cell_w, cell_h), Image.Resampling.LANCZOS)
            canvas.paste(panel, (x0, y0))
            draw.rectangle(
                [(x0, y0), (x0 + cell_w - 1, y0 + cell_h - 1)],
                outline=(209, 213, 219),
                width=2,
            )

    out_path = images_dir / "reid_comparison_grid_3x2.png"
    canvas.save(out_path)
    return out_path


if __name__ == "__main__":
    output = render()
    print(f"Wrote: {output}")
