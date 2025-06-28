# MerCBO/utils/fix_numpy_aliases.py
import numpy as _np

# restore the old names as aliases to builtins
_aliases = {
    "float":   float,
    "int":     int,
    "bool":    bool,
    "complex": complex,
    "long":    int,      # <— add this line
}

for _alias, _real in _aliases.items():
    if not hasattr(_np, _alias):
        setattr(_np, _alias, _real)
