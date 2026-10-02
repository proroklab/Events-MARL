from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent


def load_module(module_name: str, path: Path):
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_model(name: str):
    return load_module(name, PACKAGE_ROOT / "models" / f"{name}.py")


def load_scenario(file_stem: str, module_name: str):
    return load_module(module_name, PACKAGE_ROOT / "envs" / "scenarios" / f"{file_stem}.py")

