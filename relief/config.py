"""Config loading: defaults <- YAML file <- --set overrides."""
import copy
from pathlib import Path

import yaml

DEFAULTS_PATH = Path(__file__).with_name("defaults.yaml")
PRESETS_PATH = Path(__file__).with_name("cloud_presets.yaml")
TYPE_ALIASES = {"scattered": "cumulus"}


def deep_merge(base, override):
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def apply_override(cfg, expr):
    """Apply 'a.b.0.c=value' (value parsed as YAML) to a nested dict/list."""
    key, _, raw = expr.partition("=")
    value = yaml.safe_load(raw)
    parts = key.strip().split(".")
    node = cfg
    for p in parts[:-1]:
        node = node[int(p)] if isinstance(node, list) else node.setdefault(p, {})
    last = parts[-1]
    if isinstance(node, list):
        node[int(last)] = value
    else:
        node[last] = value


def load_config(path, overrides=()):
    with open(DEFAULTS_PATH) as f:
        cfg = yaml.safe_load(f)
    with open(path) as f:
        cfg = deep_merge(cfg, yaml.safe_load(f) or {})
    cfg["clouds"]["layers"] = [resolve_layer(l, cfg["clouds"]["types"])
                               for l in cfg["clouds"].get("layers") or []]
    for expr in overrides:
        apply_override(cfg, expr)
    return cfg


def resolve_layer(layer, types):
    """Cloud layer = common defaults <- type defaults <- preset <- the layer's own keys."""
    with open(PRESETS_PATH) as f:
        presets = yaml.safe_load(f)
    preset = {}
    if layer.get("preset"):
        if layer["preset"] not in presets:
            raise KeyError(f"unknown cloud preset {layer['preset']!r}; options: {', '.join(presets)}")
        preset = presets[layer["preset"]]
    t = layer.get("type") or preset.get("type") or "valley_fog"
    t = TYPE_ALIASES.get(t, t)
    if t not in types or t == "common":
        raise KeyError(f"unknown cloud type {t!r}")
    out = deep_merge(deep_merge(deep_merge(types["common"], types[t]), preset), layer)
    out["type"] = t
    return out
