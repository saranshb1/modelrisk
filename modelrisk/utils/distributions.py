"""Distribution fitting utilities."""

from __future__ import annotations

#Third-party imports
import numpy as np
import pandas as pd
from scipy import stats


SUPPORTED_DISTRIBUTIONS = {
    "normal": stats.norm,
    "lognormal": stats.lognorm,
    "gamma": stats.gamma,
    "beta": stats.beta,
    "pareto": stats.pareto,
    "exponential": stats.expon,
    "weibull": stats.weibull_min,
    "t": stats.t,
    "genpareto": stats.genpareto,
    "negative_binomial": stats.nbinom,
}


def fit_distribution(
    data: np.ndarray | pd.Series,
    distribution: str,
    **kwargs,
) -> dict:
    """Fit a named distribution to data by MLE.

    Parameters
    ----------
    data : array-like
    distribution : str
        One of the SUPPORTED_DISTRIBUTIONS keys.
    **kwargs
        Passed to scipy.stats fit() (e.g. floc=0 to fix location).

    Returns
    -------
    dict with keys: distribution, params, aic, bic, ks_statistic, ks_pvalue.
    """
    if distribution not in SUPPORTED_DISTRIBUTIONS:
        raise ValueError(
            f"Unknown distribution '{distribution}'. "
            f"Supported: {list(SUPPORTED_DISTRIBUTIONS.keys())}"
        )
    x = np.asarray(data, dtype=float)
    dist = SUPPORTED_DISTRIBUTIONS[distribution]
    params = dist.fit(x, **kwargs)

    log_likelihood = np.sum(dist.logpdf(x, *params))
    k = len(params)
    n = len(x)
    aic = 2 * k - 2 * log_likelihood
    bic = k * np.log(n) - 2 * log_likelihood

    ks_stat, ks_pval = stats.kstest(x, distribution, args=params)

    return {
        "distribution": distribution,
        "params": params,
        "log_likelihood": log_likelihood,
        "aic": aic,
        "bic": bic,
        "ks_statistic": ks_stat,
        "ks_pvalue": ks_pval,
    }


class DistributionFitter:
    """Fit and compare multiple distributions to a dataset.

    Parameters
    ----------
    distributions : list of str or None
        Distributions to try. If None, uses a standard set.

    Examples
    --------
    >>> fitter = DistributionFitter()
    >>> fitter.fit(losses)
    >>> fitter.best(criterion='aic')
    >>> fitter.comparison_table()
    """

    DEFAULT_DISTRIBUTIONS = ["normal", "lognormal", "gamma", "pareto", "exponential", "weibull"]

    def __init__(self, distributions: list[str] | None = None) -> None:
        self.distributions = distributions or self.DEFAULT_DISTRIBUTIONS
        self._results: list[dict] = []

    def fit(self, data: np.ndarray | pd.Series, **kwargs) -> DistributionFitter:
        """Fit all candidate distributions.

        Parameters
        ----------
        data : array-like of positive values.
        **kwargs : passed to each scipy.stats fit() call (e.g. floc=0).
        """
        x = np.asarray(data, dtype=float)
        self._results = []
        for dist_name in self.distributions:
            try:
                result = fit_distribution(x, dist_name, **kwargs)
                self._results.append(result)
            except Exception as e:
                self._results.append(
                    {"distribution": dist_name, "aic": np.inf, "bic": np.inf,
                    "error": str(e)}
                )
        return self

    def best(self, criterion: str = "aic") -> dict:
        """Return the best-fitting distribution by AIC or BIC.

        Parameters
        ----------
        criterion : str — 'aic' or 'bic'.
        """
        valid = [r for r in self._results if criterion in r and np.isfinite(r[criterion])]
        if not valid:
            raise RuntimeError("No distributions fitted successfully.")
        return min(valid, key=lambda r: r[criterion])

    def comparison_table(self) -> pd.DataFrame:
        """Return a DataFrame comparing all fitted distributions."""
        rows = []
        for r in self._results:
            rows.append(
                {
                    "distribution": r.get("distribution"),
                    "aic": r.get("aic", np.inf),
                    "bic": r.get("bic", np.inf),
                    "log_likelihood": r.get("log_likelihood", np.nan),
                    "ks_statistic": r.get("ks_statistic", np.nan),
                    "ks_pvalue": r.get("ks_pvalue", np.nan),
                }
            )
        return (
            pd.DataFrame(rows)
            .sort_values("aic")
            .reset_index(drop=True)
        )
