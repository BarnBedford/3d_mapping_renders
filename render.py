"""Render a shaded-relief map from a YAML config.

    python render.py configs/olympus.yaml
    python render.py configs/olympus.yaml --preview
    python render.py configs/olympus.yaml --set terrain.z_exaggeration=3 --set clouds.layers.0.density=0.5
"""
import argparse

from relief.config import load_config
from relief.render import render


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override a config value, e.g. sun.altitude_deg=25 (repeatable)")
    p.add_argument("--preview", action="store_true", help="quick 1200 px wide render")
    p.add_argument("--width", type=int, help="output width in px")
    p.add_argument("--out", help="output path (.png)")
    a = p.parse_args()

    cfg = load_config(a.config, a.set)
    if a.preview:
        cfg["output"]["width_px"] = min(cfg["output"]["width_px"], 1200)
        cfg["output"]["path"] = cfg["output"]["path"].replace(".png", "_preview.png")
    if a.width:
        cfg["output"]["width_px"] = a.width
    if a.out:
        cfg["output"]["path"] = a.out
    render(cfg)


if __name__ == "__main__":
    main()
