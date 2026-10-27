# SPDX-License-Identifier: NRC-2
# Copyright 2026-present the X8 authors. All rights reserved. See /LICENSE-NRC-2.md

"""Put this repository's root on sys.path, so ``features.*`` and ``shared.*`` import.

pytest adds the rootdir of a collected test file, which for
``features/cmt/test_cmt.py`` is ``features/cmt``. The modules import each other by
package name, so the root has to be importable instead.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
