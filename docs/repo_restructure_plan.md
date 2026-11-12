# Repo restructure: the `quant_toolkit` namespace family (PR X0)

> **Status: AGREED, NOT STARTED (2026-11-11).** Decision taken in design discussion; detail of
> the one PR that does it. It is PR X0 of [xl_api_plan.md](xl_api_plan.md) and must land alone:
> it rewrites every import in the repo. Calendar work in flight on another branch merges first;
> the import codemod below makes the rebase of anything else mechanical.

## 0. Decisions, so they are not re-litigated

1. **One import namespace, several distributions.** Everything lives under `quant_toolkit`
   (PEP 420 implicit namespace). Each portion is its own distribution with its own
   `pyproject.toml`; a root meta-package `quant-toolkit` pulls the library portions in and offers
   `excel` as an extra. Library users never install Excel code.
2. **Portions and their dependency graph** (layering is enforced by what each toml declares):

   | Distribution | Import | Depends on (internal) |
   |---|---|---|
   | `quant-toolkit-common` | `quant_toolkit.common` | — |
   | `quant-toolkit-infra` (later, when it has content) | `quant_toolkit.infra` | common |
   | `quant-toolkit-finance` | `quant_toolkit.finance` | common (+ infra later) |
   | `quant-toolkit-xl` | `quant_toolkit.xl` | finance |
   | `quant-toolkit` (meta) | — | common, finance; extra `excel` → xl |

3. **Lockstep versions.** Every portion carries the same version string and they release
   together; internal pins are `==` that version. One place holds it (`VERSION` file at the root
   read by each toml via a tiny `setuptools` dynamic hook, or a release script that rewrites all
   tomls). Independent versioning of tightly coupled portions is the cost we are not paying.
4. **No `__init__.py` at `quant_toolkit/` in any portion, ever.** A stray one hides every other
   portion silently. A test in `common` walks the repo and fails if one appears.
5. **The `quant_toolkit` shim package is dropped.** The public API is `quant_toolkit.finance`
   (and `quant_toolkit.common` for the utilities that are public: `Line1d`, interpolators,
   testing bases).
6. **Name of the utilities portion stays `common`.** "core" has been used in conversation, but
   `finance._core` is the compiled extension and `quant_toolkit.core` next to
   `quant_toolkit.finance._core` is a readability trap. Rename both or neither; neither for now.
7. **Dev flow is editable installs, not `PYTHONPATH`.** `bin/use_env.sh` installs the portions
   editable after activating the env. `PYTHONPATH` across portions still works (namespace
   portions merge across `sys.path` entries) but is no longer the documented path.
8. **Not now.** uv workspace (it wants to own the venv; the env is conda), moving tests out of the
   packages (they stay in-package, as today), splitting `_core` into a binary wheel.

## 1. Target layout

```
pyproject.toml                         quant-toolkit meta-package + ALL tool config (ruff, mypy, pytest, coverage)
VERSION                                single version string for every portion
bin/use_env.sh                         creates/activates the conda env, then runs bin/dev_install.sh
bin/dev_install.sh                     pip install -e on each portion (idempotent; CI uses it too)
bin/build_core.sh, CMakeLists.txt      C++ extension; install path moves (see §2.6)
bin/migrate_imports.py                 the import codemod (kept so other branches can run it on rebase)
packages/common/
  pyproject.toml                       quant-toolkit-common
  src/quant_toolkit/common/            today's src/common/ (incl. tests/)
packages/finance/
  pyproject.toml                       quant-toolkit-finance
  src/quant_toolkit/finance/           today's src/finance/ (incl. _core/, tests/)
  data/rates.csv                       today's src/data/rates.csv (test fixture; see §2.7)
apps/excel/
  pyproject.toml                       quant-toolkit-xl (console script qtk-xl)
  src/quant_toolkit/xl/                today's src/quant_toolkit_xl/ (incl. tests/)
  examples/                            demo workbook(s)
research/                              scripts, imports rewritten; not a distribution
docs/, conda/, pytest-reports/         unchanged locations
```

## 2. Work list

Each item names the files it touches. Order matters only where stated.

### 2.1 Move the trees (one commit, `git mv` so history follows)

- `src/common` → `packages/common/src/quant_toolkit/common`
- `src/finance` → `packages/finance/src/quant_toolkit/finance`
- `src/quant_toolkit_xl` → `apps/excel/src/quant_toolkit/xl`
- `src/quant_toolkit_xl/examples/*.xlsm` → `apps/excel/examples/`
- `src/data/rates.csv` → `packages/finance/data/rates.csv`
- delete `src/quant_toolkit/` (the shim)
- confirm `src/` is empty and remove it

### 2.2 Rewrite imports with a codemod (`bin/migrate_imports.py`)

Regex over every `*.py` and `*.ipynb` under `packages/`, `apps/`, `research/`, `docs/` and over
`*.md` under `docs/` + `README.md` + `CLAUDE.md`:

| From | To |
|---|---|
| `from finance` / `import finance` (word-bounded) | `from quant_toolkit.finance` / `import quant_toolkit.finance` |
| `from common` / `import common` | `from quant_toolkit.common` / `import quant_toolkit.common` |
| `from quant_toolkit_xl` / `import quant_toolkit_xl` | `from quant_toolkit.xl` / `import quant_toolkit.xl` |
| `from quant_toolkit import *` (the shim) | `from quant_toolkit.finance import *` |
| string literals `"finance.…"` / `"common.…"` in `COVERAGE = [...]`, `patch("…")`, registry names, docs prose | same mapping |

Counts today: 744 import lines in 168 files under `src/`, 10 files under `research/`, ~19
docs/config files mentioning module paths. The script is idempotent and committed, so a branch
opened before X0 rebases by running it once and resolving nothing by hand.

Watch-outs the codemod must handle:
- `from finance._core import datemath` and the `finance._core` string in
  `src/finance/_core/__init__.py` (module docstring + lazy import).
- `COVERAGE = ["quant_toolkit_xl.functions"]`-style strings in tests.
- `common.object.serializer` type names are **not** module paths (`@serializable("Swap")`) and
  must not be touched; verify with the frozen-fixture tests.
- The `Registry` names (`Registry(name="Pricers")`) are labels, leave them.

### 2.3 Packaging

- Root `pyproject.toml`: becomes the meta-package. `[project] name = "quant-toolkit"`,
  `dependencies = ["quant-toolkit-common==<v>", "quant-toolkit-finance==<v>"]`,
  `optional-dependencies.excel = ["quant-toolkit-xl==<v>"]`, `dev` / `cpp` / `accel` extras as
  today. Remove `[tool.setuptools]` package discovery (nothing to ship from the root).
- `packages/common/pyproject.toml`: `name = "quant-toolkit-common"`, `package-dir = {"" = "src"}`,
  `packages = find_namespace:` with `include = ["quant_toolkit.common*"]`, deps `numpy`, `scipy`,
  `pandas` (what `common` actually imports — check), `requires-python >= 3.11`.
- `packages/finance/pyproject.toml`: `name = "quant-toolkit-finance"`, same shape, deps
  `quant-toolkit-common==<v>`, `numpy`, `scipy`, `pandas`, `matplotlib` (only if still imported
  outside research — check and drop if not); extras `accel = ["numba"]`, `cpp = [...]`;
  `package-data` for `_core/*.so` and `data/rates.csv` if shipped (test fixture: probably not).
- `apps/excel/pyproject.toml`: `name = "quant-toolkit-xl"`, deps `quant-toolkit-finance==<v>`,
  `xlwings>=0.30`, `openpyxl`, `pywin32; sys_platform == "win32"`; `[project.scripts] qtk-xl =
  "quant_toolkit.xl.cli:main"` (cli lands in X4; declare the entry point then, not now);
  `package-data` for `addin/*.xlam` once it exists.
- Lockstep: `VERSION` at the root; each toml `dynamic = ["version"]` with
  `[tool.setuptools.dynamic] version = {file = ["../../VERSION"]}` (setuptools reads relative to
  the toml; verify the `../../` form builds, otherwise a 10-line `bin/bump_version.py` that
  rewrites all four tomls).

### 2.4 Tool config (root `pyproject.toml`)

- `[tool.pytest.ini_options]`: drop `pythonpath = ["src"]`; `testpaths` =
  `packages/common/src/quant_toolkit/common/tests`, `packages/finance/src/quant_toolkit/finance/tests`,
  `apps/excel/src/quant_toolkit/xl/tests`; add `--import-mode=importlib`; `--cov=quant_toolkit`
  replaces the two `--cov` flags.
- `[tool.coverage.run]`: `omit` patterns update to `*/quant_toolkit/*/tests/*`,
  `*/quant_toolkit/finance/_core/*`; `source = ["quant_toolkit"]`.
- `[tool.ruff]`: `include = ["packages/**/*.py", "apps/**/*.py", "research/**/*.py"]`.
- `[tool.mypy]` (new; there is none today): `namespace_packages = true`,
  `explicit_package_bases = true`, `mypy_path = "packages/common/src:packages/finance/src:apps/excel/src"`.
- Performance expects (`tests/performance/expects/`) are located relative to the test file; verify
  the loader still resolves after the move.

### 2.5 Environment and scripts

- `bin/dev_install.sh` (new): `pip install --no-deps -e packages/common -e packages/finance -e apps/excel`
  (the `--no-deps` keeps conda in charge of numpy/scipy; the Excel portion installs on Linux too
  because `xlwings` is guarded by the `_xw` shim and `pywin32` is a win32 marker). Prints the
  resolved `quant_toolkit.__path__` so a broken namespace is visible immediately.
- `bin/use_env.sh`: after `conda activate`, and again after `--prune`, call `bin/dev_install.sh`.
  Keep it sourced-only as today.
- `conda/pricing/pricing-*.yml`: unchanged (no `-e` entries in the yml; relative-path semantics
  there are a known trap).
- Windows venv for Excel testing: `pip install -e <unc>\packages\common -e <unc>\packages\finance -e <unc>\apps\excel`;
  `UDF Modules` becomes `quant_toolkit.xl.functions`; the demo workbook's `xlwings.conf` sheet updated.

### 2.6 C++ extension

- `CMakeLists.txt`: source `packages/finance/src/quant_toolkit/finance/_core/cpp/datemath.cpp`;
  `install(... DESTINATION packages/finance/src/quant_toolkit/finance/_core)`; comments.
- `bin/build_core.sh`: comments and the "lands in" path.
- `_core/__init__.py`: lazy import path `quant_toolkit.finance._core`.
- Rebuild and run the `compiled` schedule tests to confirm the `.so` is found.

### 2.7 Data and fixtures

- `test_historical_fixings.py` locates `rates.csv` with `Path(__file__).parents[4] / "data"`;
  after the move the depth changes. Replace with a `finance_data_dir()` helper in
  `quant_toolkit.finance.tests` (or `common.testing`) that resolves `packages/finance/data`, so
  no test does parent-counting again.
- Expects files and frozen serialization fixtures move with their test trees; nothing in them
  references module paths (type names are stable strings) — the fixture tests are the check.

### 2.8 Docs and guidance

- `CLAUDE.md`: package layout, commands (`source bin/use_env.sh pricing` now installs the
  portions; `pytest` unchanged; `bin/build_core.sh` unchanged), module paths in the Architecture
  section (`quant_toolkit.finance.markets.curves` etc.), the test-infrastructure paths.
- `README.md`: folder structure section, Excel link → `apps/excel/README.md`, install lines.
- `docs/*.md` (19 files): module paths via the codemod; `docs/xl_api_plan.md` §1 already matches.
- `apps/excel/README.md`: import names, `UDF Modules` value, install commands.
- Memory note for the assistant: the run-scripts memory (`PYTHONPATH=src ...`) is replaced by
  "activate the env; portions are editable-installed".

### 2.9 Guards (new tests, in `common`)

- No `__init__.py` at any `*/src/quant_toolkit/` directory in the repo.
- Every portion's `pyproject.toml` declares the same version as `VERSION`.
- Import layering: `quant_toolkit.common` imports nothing from `quant_toolkit.finance` or
  `quant_toolkit.xl` (walk the AST of the portion; cheap, no import-linter dependency).

## 3. Verification checklist (all green before merge)

```bash
source bin/use_env.sh pricing            # creates/activates + editable installs
python -c "import quant_toolkit; print(quant_toolkit.__path__)"   # lists all three portion roots
python -c "import quant_toolkit.finance, quant_toolkit.common, quant_toolkit.xl"
bin/build_core.sh && python -c "from quant_toolkit.finance._core import is_available; print(is_available())"
pytest                                   # full suite incl. the three guards, coverage >= 80
pytest -m performance --no-cov -s        # expects still resolve
ruff check . && ruff format --check packages apps research
mypy packages apps                       # namespace config in place
pip install packages/common packages/finance apps/excel   # non-editable wheels build and install into a scratch venv
python research/sofr_curve_calibration.py                 # a research script still runs
```

Plus the Excel smoke: reinstall the Windows venv editable, re-import functions in the demo
workbook (module name changed), recalc, PV unchanged at 346,756.20.

## 4. Order and coordination

1. Carlos's calendar branch merges to `main` first (it touches `finance.dates`, which moves).
2. Commit today's working-tree changes (the `qMarket` parameter fix, guard test, README rows,
   the two plan docs) on their own branch and merge, so X0 is pure restructure.
3. X0 on `feature/carlos/namespace_family`: §2.1 move, §2.2 codemod, §2.3–2.9 in any order,
   §3 checklist, PR, merge the same day. Nothing else opens against `main` while it is up.
4. Any branch opened before X0 rebases with `python bin/migrate_imports.py` and a `git mv` of
   whatever new files it added under `src/`.

Rough effort: one day including the Excel re-verification.

## 5. Open points

1. Does `packages/finance/data/rates.csv` ship in the wheel or stay a repo-only test fixture?
   Planned: repo-only, located through the helper in §2.7.
2. `VERSION` file read by each toml versus a bump script. Planned: try the file first.
3. Whether `research/` should become `quant_toolkit.research` (a portion with no tests, excluded
   from coverage) or stay loose scripts. Planned: loose scripts for now.
