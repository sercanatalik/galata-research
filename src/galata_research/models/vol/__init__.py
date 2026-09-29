"""Volatility models: the GARCH family in-sample, with each model's own persistence.

    f = gr.models.vol.fit(returns, model="gjr", dist="t")
    gr.models.vol.table([f, ...])      # one row per fit
    gr.models.vol.news_impact(f, z)    # Engle and Ng's curve
    gr.models.vol.diagnose(f)          # Ljung-Box on z and z², ARCH-LM
    gr.models.vol.walk_forward(returns, model="garch", split=t, every=5, horizons=[1, 7])
    gr.models.vol.har(gr.timeseries.realized_from(bars_4h, "1d"), model="harq", split=t, horizons=[1, 7])
    gr.models.vol.har(gr.models.vol.realized_jumps(bars_5m, "1d"), model="hartcj", split=t)  # jumps apart
"""

from .._arch import DISTS
from .carr import carr
from .garch import MODELS, Fit, arch_lm, diagnose, expectation, fit, kappa, ljung_box, news_impact, segmented, table
from .har import har
from .jumps import realized_jumps
from .target import economics, es_t, estimation_target, target, trials
from .walk import walk_forward

__all__ = ["DISTS", "MODELS", "Fit", "arch_lm", "carr", "diagnose", "es_t", "estimation_target", "expectation", "fit", "har", "kappa", "ljung_box", "news_impact", "realized_jumps", "segmented", "table", "target", "trials", "economics", "walk_forward"]
