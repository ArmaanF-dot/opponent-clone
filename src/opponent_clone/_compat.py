"""maia2 imports pyzstd (only for its own training-data decompression). Its compiled module fails to load on
some systems (e.g. Kaggle), so stub it when that happens. We never call it."""
import sys
import types


def stub_pyzstd() -> None:
    try:
        import pyzstd  # noqa: F401
    except Exception:
        sys.modules["pyzstd"] = types.ModuleType("pyzstd")
