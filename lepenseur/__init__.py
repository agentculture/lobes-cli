"""lepenseur — the local thinking agent of the Culture mesh."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _v

try:
    __version__ = _v("lepenseur")
except PackageNotFoundError:  # editable install without metadata
    __version__ = "0.0.0+local"

__all__ = ["__version__"]
