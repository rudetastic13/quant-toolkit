#!/usr/bin/env python
"""Set the lockstep version: rewrite VERSION and every portion's pyproject.toml.

    python bin/bump_version.py 0.2.0

Each portion carries the same version string and pins its internal dependencies with ``==`` that version, so
all of them move together.  ``test_repo_layout`` fails if any file disagrees with VERSION.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOMLS = [ROOT / "pyproject.toml", *sorted(ROOT.glob("packages/*/pyproject.toml")), *sorted(ROOT.glob("apps/*/pyproject.toml"))]
VERSION_LINE = re.compile(r'^(version\s*=\s*")[^"]+(")', re.MULTILINE)
INTERNAL_PIN = re.compile(r'("quant-toolkit(?:-[a-z]+)?==)[^"]+(")')


def main() -> int:
    if len(sys.argv) != 2 or not re.fullmatch(r"\d+\.\d+\.\d+(?:[.\-+]\w+)*", sys.argv[1]):
        print(__doc__)
        return 1
    version = sys.argv[1]
    (ROOT / "VERSION").write_text(version + "\n")
    for toml in TOMLS:
        text = toml.read_text()
        text = VERSION_LINE.sub(rf"\g<1>{version}\g<2>", text, count=1)
        text = INTERNAL_PIN.sub(rf"\g<1>{version}\g<2>", text)
        toml.write_text(text)
        print(f"updated {toml.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
