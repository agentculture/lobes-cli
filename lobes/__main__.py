"""Allow running lobes as ``python -m lobes``."""

import sys

from lobes.cli import main

if __name__ == "__main__":
    sys.exit(main())
