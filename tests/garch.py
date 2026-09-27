import re
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from arch import arch_model
from conftest import utc

import galata_research as gr
from galata_research import Refused

vol = gr.models.vol
DAY = timedelta(days=1)
SRC = Path(__file__).parent.parent / "src" / "galata_research"


def _frame(values, *, ticker="BTC"):
    t0 = utc("2020-01-01T00:00")
    n = len(values)
    return pl.DataFrame(
        {"ticker": ticker, "ts": [t0 + i * DAY for i in range(n)], "close_ts": [t0 + (i + 1) * DAY for i in range(n)], "return": values},
        schema_overrides={"return": pl.Float64},
    )


@pytest.fixture(scope="module", name="simulated")
def _simulated():
    # GARCH(1,1)-t on the ×100 scale: ω 0.05, α 0.08, β 0.9, ν 5, seeded; returned in return units.
    np.random.seed(11)
    sim = arch_model(None, dist="t").simulate([0.0, 0.05, 0.08, 0.9, 5.0], 5000, burn=500)
    return _frame((sim["data"].to_numpy() / 100).tolist())


def a_garch_t_recovers_its_simulation(simulated):
    f = vol.fit(simulated, model="garch", dist="t")
    assert f.params["alpha[1]"] == pytest.approx(0.08, abs=0.03)
    assert f.params["beta[1]"] == pytest.approx(0.9, abs=0.03)
    assert f.params["nu"] == pytest.approx(5.0, abs=1.5)
    assert f.persistence == pytest.approx(f.params["alpha[1]"] + f.params["beta[1]"])
    assert f.half_life == pytest.approx(np.log(0.5) / np.log(f.persistence))


def the_parameter_names_per_model(simulated):
    want = {
        "garch": ["mu", "omega", "alpha[1]", "beta[1]", "nu"],
        "gjr": ["mu", "omega", "alpha[1]", "gamma[1]", "beta[1]", "nu"],
        "egarch": ["mu", "omega", "alpha[1]", "gamma[1]", "beta[1]", "nu"],
        "aparch": ["mu", "omega", "alpha[1]", "gamma[1]", "beta[1]", "delta", "nu"],
        "figarch": ["mu", "omega", "phi", "d", "beta", "nu"],
    }
    for model, names in want.items():
        assert list(vol.fit(simulated, model=model, dist="t").params) == names, model


def a_frame_of_two_tickers_is_refused(simulated):
    both = pl.concat([simulated, simulated.with_columns(pl.lit("ETH").alias("ticker"))])
    with pytest.raises(Refused, match="BTC, ETH"):
        vol.fit(both)


def a_sample_under_min_obs_is_refused(simulated):
    with pytest.raises(Refused, match="264 returns, under min_obs=500"):
        vol.fit(simulated.head(264))


def an_unknown_model_is_refused(simulated):
    with pytest.raises(Refused, match="model='tgarch'"):
        vol.fit(simulated, model="tgarch")


def the_kappa_of_a_symmetric_distribution():
    assert vol.kappa("t", {"nu": 5.0}) == pytest.approx(0.5, abs=1e-6)
    assert vol.kappa("normal", {}) == pytest.approx(0.5, abs=1e-6)


def the_kappa_of_a_skewed_distribution():
    from arch.univariate import SkewStudent

    draws = SkewStudent(seed=np.random.default_rng(3)).simulate([5.0, -0.3])(400_000)
    monte_carlo = float(np.mean(np.where(draws < 0, draws**2, 0.0)))
    got = vol.kappa("skewt", {"eta": 5.0, "lambda": -0.3})
    assert got == pytest.approx(monte_carlo, abs=1e-2)
    assert abs(got - 0.5) > 0.05


def the_egarch_persistence_is_beta(simulated):
    f = vol.fit(simulated, model="egarch", dist="t")
    assert f.persistence == f.params["beta[1]"]
    assert f.sigma_bar is None


def a_hole_is_bridged_and_marked(simulated):
    values = simulated["return"].to_list()[:800]
    values[0] = None
    values[400] = None
    f = vol.fit(_frame(values))
    assert f.nobs == 798
    flagged = f.series.filter(pl.col("after_gap"))
    assert flagged.height == 1
    assert flagged["ts"].item() == simulated["ts"][401]


def the_gjr_curve_leans_on_bad_news(simulated):
    f = vol.fit(simulated, model="gjr", dist="t")
    curve = dict(vol.news_impact(f, [-2.0, 2.0]).iter_rows())
    assert (curve[-2.0] > curve[2.0]) == (f.params["gamma[1]"] > 0)


def the_garch_curve_is_symmetric(simulated):
    curve = vol.news_impact(vol.fit(simulated, model="garch"), [-2.0, 2.0])["sigma2"].to_list()
    assert curve[0] == pytest.approx(curve[1])


def a_figarch_has_no_news_impact_curve(simulated):
    with pytest.raises(Refused, match="whole past"):
        vol.news_impact(vol.fit(simulated, model="figarch"), [0.0])


def the_ljung_box_by_hand():
    x = [0.3, -1.2, 0.8, 0.1, -0.4, 1.5, -0.9, 0.2]
    n, d = len(x), np.array(x) - np.mean(x)
    rho = [float(d[k:] @ d[:-k]) / float(d @ d) for k in (1, 2)]
    assert vol.ljung_box(x, 2) == pytest.approx(n * (n + 2) * (rho[0] ** 2 / (n - 1) + rho[1] ** 2 / (n - 2)))


def the_arch_lm_agrees_with_arch(simulated):
    res = arch_model(simulated["return"].to_numpy() * 100, dist="t").fit(disp="off")
    theirs = res.arch_lm_test(lags=10, standardized=True)
    ours, p = vol.arch_lm(res.std_resid, 10)
    assert ours == pytest.approx(float(theirs.stat), rel=1e-6)
    assert p == pytest.approx(float(theirs.pval), rel=1e-6)


def a_white_noise_passes_every_diagnostic():
    rng = np.random.default_rng(5)
    f = vol.fit(_frame((rng.standard_t(6, 3000) / 100).tolist()), model="garch", dist="t")
    assert (vol.diagnose(f)["p_value"] > 0.01).all()


def a_fits_types_are_polars_and_python(simulated):
    f = vol.fit(simulated, model="gjr", dist="skewt")
    assert isinstance(f.series, pl.DataFrame)
    assert all(type(v) is float for v in [*f.params.values(), *f.std_err.values()])
    assert isinstance(vol.table([f]), pl.DataFrame) and isinstance(vol.diagnose(f), pl.DataFrame)


def no_module_but_arch_touches_pandas():
    touching = sorted(
        str(p.relative_to(SRC)) for p in SRC.rglob("*.py") if re.search(r"^\s*(import pandas|from pandas)", p.read_text(), re.M)
    )
    assert touching == [], touching
    # _arch reaches pandas only through arch's return values: it imports none itself.


def the_core_imports_without_arch():
    code = (
        "import sys; sys.modules['arch'] = None\n"
        "import galata_research as gr\n"
        "try:\n    gr.models\nexcept gr.Refused as r:\n    print('refused:', r)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert "refused:" in out and "models extra" in out and "uv sync --extra models" in out


def the_models_load_on_use():
    assert "models" in dir(gr)
    assert gr.models.vol is vol


def the_gjr_persistence_weights_gamma_by_kappa(simulated):
    # Guard: with P(z<0) or 0.5 in place of κ, a skewed fit's persistence would be wrong.
    f = vol.fit(simulated, model="gjr", dist="skewt")
    p = f.params
    assert f.persistence == pytest.approx(p["alpha[1]"] + p["beta[1]"] + p["gamma[1]"] * vol.kappa("skewt", p))
