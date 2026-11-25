"""Render an area once per cloud preset and tile the results into a contact sheet.

    python cloud_gallery.py configs/arran.yaml
    python cloud_gallery.py configs/arran.yaml --presets faint_mist,mountain_cap --width 800
"""
import argparse
import copy
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageFont

from relief.config import PRESETS_PATH, load_config, resolve_layer
from relief.render import render


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config")
    p.add_argument("--presets", help="comma-separated preset names (default: all)")
    p.add_argument("--width", type=int, default=600, help="width of each tile in px")
    p.add_argument("--cols", type=int, default=4)
    a = p.parse_args()

    with open(PRESETS_PATH) as f:
        names = list(yaml.safe_load(f))
    if a.presets:
        names = [n.strip() for n in a.presets.split(",")]

    base = load_config(a.config)
    base["output"].update(width_px=a.width, geotiff=False, save_layers=False)
    out_dir = Path(base["output"]["path"]).with_suffix("").parent / "gallery"
    stem = Path(base["output"]["path"]).stem

    tiles = []
    for name in ["no_clouds"] + names:
        cfg = copy.deepcopy(base)
        cfg["clouds"]["enabled"] = name != "no_clouds"
        cfg["clouds"]["layers"] = [] if name == "no_clouds" else [resolve_layer({"preset": name}, cfg["clouds"]["types"])]
        cfg["output"]["path"] = str(out_dir / f"{stem}_{name}.png")
        print(f"--- {name}")
        tiles.append((name, Image.open(render(cfg))))

    w, h = tiles[0][1].size
    cols = min(a.cols, len(tiles))
    rows = -(-len(tiles) // cols)
    sheet = Image.new("RGB", (cols * w, rows * h), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", max(14, w // 22))
    except OSError:
        font = ImageFont.load_default()
    for i, (name, im) in enumerate(tiles):
        x, y = (i % cols) * w, (i // cols) * h
        sheet.paste(im, (x, y))
        draw.rectangle([x, y, x + w - 1, y + h - 1], outline="white", width=2)
        draw.text((x + 12, y + 10), name, fill="white", font=font, stroke_width=3, stroke_fill="black")
    path = Path(base["output"]["path"]).with_name(f"{stem}_cloud_gallery.png")
    sheet.save(path)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
