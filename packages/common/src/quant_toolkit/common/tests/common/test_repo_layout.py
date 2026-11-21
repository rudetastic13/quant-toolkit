"""Repo-level guards for the quant_toolkit namespace family.

They read the source checkout, so they skip when the package is installed without it (a wheel).
"""
import ast
import re
import tomllib
import unittest
from pathlib import Path

from quant_toolkit.common.testing import UnitTest

HERE = Path(__file__).resolve()


def _repo_root() -> Path | None:
    for parent in HERE.parents:
        if (parent / "VERSION").is_file() and (parent / "packages").is_dir():
            return parent
    return None


ROOT = _repo_root()


@unittest.skipIf(ROOT is None, "not running from a source checkout")
class TestRepoLayout(UnitTest):
    def test_no_init_at_namespace_root(self):
        # a stray quant_toolkit/__init__.py silently hides every other portion
        offenders = [p for p in ROOT.rglob("__init__.py") if p.parent.name == "quant_toolkit" and p.parent.parent.name == "src"]
        self.assertEqual(offenders, [])

    def test_versions_match_VERSION_file(self):
        version = (ROOT / "VERSION").read_text().strip()
        tomls = [ROOT / "pyproject.toml", *ROOT.glob("packages/*/pyproject.toml"), *ROOT.glob("apps/*/pyproject.toml")]
        self.assertGreaterEqual(len(tomls), 4)
        for toml in tomls:
            with self.subTest(toml=str(toml.relative_to(ROOT))):
                data = tomllib.loads(toml.read_text())
                self.assertEqual(data["project"]["version"], version)
                requirements = list(data["project"].get("dependencies", []))
                for extra in data["project"].get("optional-dependencies", {}).values():
                    requirements += extra
                for requirement in requirements:
                    match = re.match(r"quant-toolkit(?:-[a-z]+)?\s*==\s*([^\s;]+)", requirement)
                    if match:
                        self.assertEqual(match.group(1), version, requirement)

    def test_layering_between_portions(self):
        # common imports neither finance nor xl; finance does not import xl
        forbidden = {
            "common": ("quant_toolkit.finance", "quant_toolkit.xl"),
            "finance": ("quant_toolkit.xl",),
        }
        roots = {"common": ROOT / "packages/common/src", "finance": ROOT / "packages/finance/src"}
        for portion, banned in forbidden.items():
            for path in roots[portion].rglob("*.py"):
                if "tests" in path.parts:
                    continue
                for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                    names = []
                    if isinstance(node, ast.Import):
                        names = [alias.name for alias in node.names]
                    elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                        names = [node.module]
                    for name in names:
                        with self.subTest(file=str(path.relative_to(ROOT)), imports=name):
                            self.assertFalse(any(name == b or name.startswith(b + ".") for b in banned))
