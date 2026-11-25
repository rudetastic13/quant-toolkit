"""``QuantToolkit.Risk.*`` worksheet functions (xlOil): risk to the quotes of a chosen set of instruments.

A ``RiskTemplate`` is a list of quote handles (the ``QuantToolkit.Solver`` quote functions) to
report risk against. It is a view only: the curve, the PVs and the calibration are untouched,
so one calibrated curve can be read through several templates. ``RiskReport`` re-expresses a
portfolio's pillar risk as DV01 to each template quote, with the hedge and reconciliation lines.
"""
from __future__ import annotations

import numpy as np

from quant_toolkit.finance.pricing import calibration as cal
from quant_toolkit.finance.pricing import risk as lib
from quant_toolkit.finance.pricing.risk.basis import LeastSquaresBasis, SchemeBasis, TentBasis

from quant_toolkit.xl.solver import CalibrationState, market_of
from quant_toolkit.xl.core import book, handles
from quant_toolkit.xl.core.convert import given
from quant_toolkit.xl.core.register import xl_func

_BASES = {"tent": TentBasis, "leastsquares": LeastSquaresBasis, "scheme": SchemeBasis}


@xl_func("Risk")
def RiskTemplate(Instruments, Labels=None, Basis=None, Name=None):  # noqa: N802, N803
    """The set of quotes to report risk against.

    Returns a RiskTemplate handle for RiskReport and Calculate. A template is a view: it changes
    no PV and does not recalibrate, so one curve can be read through several templates.

    Parameters
    ----------
    Instruments : range
        Quote handles from the Solver quote functions, in maturity order per curve, or a Calibration handle
        for its own quotes. The quoted rate is not used.
    Labels : range, optional
        One label per quote, e.g. "SOFR 5Y". Curve key and pillar date if omitted.
    Basis : str, optional
        How risk between template nodes is attributed: "Tent" (default), "LeastSquares" or "Scheme".
    Name : str, optional
        A name for the template, for your own reference.

    Returns
    -------
    str
        A RiskTemplate handle ("RiskTemplate;;;9fc97b4464ac").

    Raises
    ------
    TypeError
        If an element is not a quote.
    ValueError
        If Basis is not recognised, or the number of labels differs from the number of quotes.

    Notes
    -----
    Problems with the template itself (nodes out of order, more nodes than the curve has
    pillars, a node finer than the curve) are raised when it is first used against a market.
    """
    quotes: list = []
    for item in handles.flatten(Instruments):
        if isinstance(item, CalibrationState):
            quotes.extend(item.quotes)
        elif isinstance(item, list):
            quotes.extend(item)
        elif given(item):
            quotes.append(item)
    bad = [q for q in quotes if not isinstance(q, cal.CalibrationInstrument)]
    if bad:
        raise TypeError(f"{bad[0]!r} is not a quote: build the range from the QuantToolkit.Solver quote functions")
    labels = tuple(str(v) for v in handles.flatten(Labels) if given(v)) if Labels is not None else None
    basis_name = str(Basis).strip().lower().replace("_", "") if given(Basis) else "tent"
    if basis_name not in _BASES:
        raise ValueError(f"unknown basis {Basis!r}: use Tent, LeastSquares or Scheme")
    template = lib.RiskTemplate(
        tuple(quotes), basis=_BASES[basis_name](), labels=labels or None, name=str(Name) if given(Name) else ""
    )
    return handles.store("RiskTemplate", template, Instruments, labels, basis_name, Name)


@xl_func("Risk")
def RiskReport(Portfolio, Market, RiskTemplate=None):  # noqa: N802, N803
    """Portfolio risk expressed per template quote, with reconciliation lines.

    Parameters
    ----------
    Portfolio : str or range
        A Portfolio handle, or swap handles (compiled on the spot).
    Market : str
        A Calibration handle.
    RiskTemplate : str, optional
        A RiskTemplate handle. The quotes the curve was calibrated to if omitted.

    Returns
    -------
    table
        A header row, then per template quote: Quote (label), Pillar (date), Kind, Rate (the
        quote the curve implies), DV01 (portfolio PV change for +1bp on that quote), Hedge
        (notional of that quote's instrument that flattens it). Then three rows: all quotes
        +1bp, all pillars +1bp, and the largest pillar risk the template leaves unexplained.

    Notes
    -----
    The two parallel lines differ by design: moving every quote by 1bp is not the same curve
    move as moving every zero rate by 1bp. A template coarser than the portfolio leaves
    unexplained risk at the pillars between its nodes.
    """
    state = handles.load(Market)
    market, _ = market_of(state)
    template = book.template_of(RiskTemplate, state)
    engine = lib.RiskEngine(book.book_of(Portfolio).program, market, template=template)
    risk = engine.quote_risk()
    bound = engine.bound()
    quote_dv01 = risk.dv01.sum(axis=0)
    hedge = bound.hedge(risk).sum(axis=0)
    pillars = book.serials([h.pillar_date for h in template.instruments])
    unexplained = np.abs(risk.unexplained.sum(axis=0)).max(initial=0.0) * book.BP

    rows: list[list] = [["Quote", "Pillar", "Kind", "Rate", "DV01", "Hedge"]]
    for k, label in enumerate(risk.labels):
        rows.append(
            [label, pillars[k], risk.kinds[k].name, float(bound.implied[k]), float(quote_dv01[k]), float(hedge[k])]
        )
    rows.append(["Total (all quotes +1bp)", "", "", "", float(quote_dv01.sum()), ""])
    rows.append(["Pillar DV01 (all pillars +1bp)", "", "", "", float(risk.pillar_parallel.sum() * book.BP), ""])
    rows.append(["Unexplained (largest pillar)", "", "", "", float(unexplained), ""])
    return np.array(rows, dtype=object)
