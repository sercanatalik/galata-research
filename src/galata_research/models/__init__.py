"""Fitted models over the record: the `[models]` extra (arch, scipy, numpy).

    gr.models.vol.fit(returns, model="gjr", dist="t")   # a Fit, in polars and plain Python
    gr.models.evaluate.scorecard(aligned, benchmark="ewma")  # forecasts scored

Loaded on first use, so the core never needs numpy. pandas, which arch
returns, stays inside `_arch` and never reaches a caller.
"""

from .._errors import Refused

try:
    import arch  # noqa: F401
    import scipy  # noqa: F401
except ImportError as missing:
    raise Refused(f"gr.models needs the models extra ({missing.name} is missing): uv sync --extra models") from None

from . import discovery, evaluate, intraday, vol

__all__ = ["discovery", "evaluate", "intraday", "vol"]
