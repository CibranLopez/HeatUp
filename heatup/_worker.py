"""Child-process entry point (``python -m heatup._worker <step> <phase_dir> ...``).

A separate module so that ``python -m`` does not re-execute ``heatup.runtime`` after the package
``__init__`` has already imported it (runpy RuntimeWarning).
"""
import sys

from heatup.runtime import main

if __name__ == "__main__":
    sys.exit(main())
