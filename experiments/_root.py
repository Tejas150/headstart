"""Anchor every experiment to the repo root.

These scripts were written flat in the root, so they reference `models/` and
write their output `.wav` files with paths relative to it, and two of them
`import server` directly. Moving them into this folder breaks both.

The alternative was rewriting the model path in ten files and adding a
sys.path shim to two more -- twelve edits, each a chance to change a script
that produced a number the README quotes. Anchoring the working directory
instead keeps every path inside them literally correct and keeps the diff to
one import line per file, which is a diff you can read in full.

Import it first, before anything that touches a path:

    import _root  # noqa: F401  -- chdir to repo root

The tradeoff worth naming: this makes the scripts depend on being run from a
checkout rather than being importable from anywhere. That is what they are --
one-shot measurements whose output is a number in the README, not a library.
The server itself takes the other route and reads HEADSTART_MODEL from the
environment, because it does have to run somewhere else, namely a container.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

os.chdir(ROOT)

# So `import server` resolves from experiments/ the same way it did from root.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
