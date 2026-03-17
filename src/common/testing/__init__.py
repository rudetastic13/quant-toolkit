"""Types and reusable methods for testing"""
import re
from unittest import TestCase, skip, skipIf
import pytest
import pandas as pd

@pytest.mark.unit
class UnitTest(TestCase):
    COVERAGE = []

@pytest.mark.integration
class IntegrationTest(TestCase):
    COVERAGE = []

@pytest.mark.performance
class PerformanceTest(TestCase):
    COVERAGE = []

# region, Markdown utilities
_RE_COL_HEADER = r"^(.+) \((.+)\)$"

def dataframe_to_markdown(df: pd.DataFrame, precision=12, col_type_header: bool = True) -> str:
    df_copy = df.copy()

    if col_type_header:
        df_copy.columns = [
            f"{col} ({dtype})" for col, dtype in zip(df_copy.columns, df_copy.dtypes)
        ]
    else:
        df.copy_columns = [
            f"{col}" for col in df_copy.columns
        ]

    # to Markdown
    markdown = df_copy.to_markdown(index=False, floatfmt=f".{precision}f")

    return markdown

def _parse_header(header: str) -> tuple[str, str]:
    match = re.match(_RE_COL_HEADER, header.strip())
    return match.group(1), match.group(2)

def dataframe_from_markdown(markdown: str) -> pd.DataFrame:
    lines = markdown.strip().split("\n")
    headers, dtypes = zip(
        *[_parse_header(col) for col in lines[0].strip("|").split("|")]
    )

    data = []
    for line in lines[2:]:
        values = [val.strip() for val in line.strip("|").split("|")]
        row = dict(zip(headers, values))
        data.append(row)

    # build dataframe
    df = pd.DataFrame(data, columns=headers)
    for col, dtype in zip(headers, dtypes):
        df[col] = df[col].astype(dtype)

    return df

# endregion
