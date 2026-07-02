"""Define rate index and funding index and singleton of it"""
from finance.dates import Term
from dataclasses import dataclass, field


@dataclass
class RateIndex:
    """Class representing a rate index, such as LIBOR, EURIBOR, etc."""
    currency: str
    name: str
    tenor: str
    _term: Term = field(init=False)
    _config: dict = field(default_factory=dict, init=False)

    def __post_init__(self):
        self._term = Term.from_str(self.tenor)

    @property
    def term(self) -> Term:
        return self._term

    @property
    def config(self) -> dict:
        return self._config

    def __getitem__(self, key: str):
        return self._config[key]

@dataclass
class FundingIndex:
    """Class representing a funding index/CSA discount basis, such as SOFR-OIS discounting."""
    currency: str
    name: str
    _config: dict = field(default_factory=dict, init=False)

    @property
    def config(self) -> dict:
        return self._config

    def __getitem__(self, key: str):
        return self._config[key]