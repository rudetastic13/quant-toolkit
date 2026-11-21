#!/usr/bin/env bash
#
# Editable-install every portion of the quant_toolkit family into the active environment.
# Idempotent; CI uses it too.  --no-deps leaves numpy/scipy/pandas to the caller (conda or pip) ...
# except when run with --deps, which also installs the dev extras (tests, lint, notebooks).
#
# Usage:
#     bin/dev_install.sh [--deps]
set -euo pipefail
cd "$(dirname "$0")/.."

if [ "${1:-}" = "--deps" ]; then
  pip install -e packages/common -e packages/finance -e ".[dev,notebook,accel]"
else
  pip install --no-deps -e packages/common -e packages/finance
fi

# a broken namespace shows up immediately: all portion roots must be listed
python - <<'PY'
import quant_toolkit
paths = list(quant_toolkit.__path__)
print("quant_toolkit.__path__:")
for p in paths:
    print("  ", p)
PY
