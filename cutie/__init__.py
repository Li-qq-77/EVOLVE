"""
Compatibility package for this repository layout.

The original CUTIE/EVOLVE code imports modules as `cutie.model...` and
`cutie.inference...`. In this local reproduction tree, those folders live at
the project root (`model/`, `inference/`, `dataset/`). This shim keeps the
upstream import paths working without rewriting every source file.
"""
from importlib import import_module
import sys

for _name in ("model", "inference", "dataset"):
    try:
        _module = import_module(_name)
        sys.modules[f"{__name__}.{_name}"] = _module
        setattr(sys.modules[__name__], _name, _module)
    except Exception:
        # Some optional packages may have heavy dependencies; they can be
        # imported later when actually needed.
        pass
