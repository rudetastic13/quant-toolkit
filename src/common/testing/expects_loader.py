"""Functions and types to support loading regressiion tests"""
from typing import Protocol
import importlib
from io import IOBase
import pandas as pd
from common.testing import dataframe_from_markdown, dataframe_to_markdown

class ExpectsLoader(Protocol):
    def load(self) -> dict[str, pd.DataFrame]: ...

    def update(self, test_name: str, expects: dict[str, pd.DataFrame]) -> None: ...

class ModuleLoader(ExpectsLoader):
    def __init__(self, module_name: str):
        self.module_name = module_name

    def load(self) -> dict[str, pd.DataFrame]:
        mod = importlib.import_module(self.module_name)
        names = dir(mod)
        results = {}
        for name in names:
            if not name.startswith("__") and not name.endswith("__"):
                results[name] = dataframe_from_markdown(getattr(mod, name))
        return results

    def update(self, test_name: str, expects: dict[str, pd.DataFrame], stream: IOBase = None) -> None:
        content = f'"""\n{test_name} expects - Regression Test\n"""'
        for name, df in sorted(expects.items()):
            md = dataframe_to_markdown(df)
            content += f"\n\n{name} = '''\n{md}\n'''\n"

        if stream is not None:
            stream.write(content)
        else:
            module = importlib.import_module(self.module_name)
            with open(module.__file__, "w") as f:
                f.write(content)
