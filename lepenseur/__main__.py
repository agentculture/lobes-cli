"""Allow running lepenseur as ``python -m lepenseur``."""

import sys

from lepenseur.cli import main

if __name__ == "__main__":
    sys.exit(main())
