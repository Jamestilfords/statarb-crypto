"""
crypto_statarb.py

Clean, reusable utilities for the Crypto Statistical Arbitrage notebook:
- Cross-sectional long/short portfolio construction (momentum, reversal)
- Daily rebalancing on 4H bars with 1-bar lag (no look-ahead)
- Turnover + bps transaction cost model
- Sweeps (momentum L, reversal H), stress tests, train/test, bucket diagnostic
- Optional: liquidity (volume-rank) filter

Note:
- This module intentionally avoids hard-coding "best_H" / "best_L".
- Data fetching from exchanges is intentionally left minimal; in many cases you will
  want to cache OHLCV locally (parquet/csv) for reproducible results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Portfolio construction
# -----------------------------
def make_long_short_weights(score: pd.DataFrame, q: float = 0.2, min_assets: int = 5) -> pd.DataFrame:
    """
    Cross-sectional long/short:
      - Long top q fraction by score
      - Short bottom q fraction by score
      - Equal weight within each side
    Output weights are per-row (time) and per-column (asset).
    """
    w = pd.DataFrame(0.0, index=score.index, columns=score.columns)

    for t in score.index:
        s = score.loc[t].dropna()
        n = len(s)
        if n < min_assets:
            continue

        k = max(1, int(n * q))  # always at least 1 long + 1 short
        longs = s.sort_values(ascending=False).head(k).index
        shorts = s.sort_values(ascending=True).head(k).index

        w.loc[t, longs] =  1.0 / k
        w.loc[t, shorts] = -1.0 / k

    return w


def make_long_short_weights_masked(
    score: pd.DataFrame,
    tradable: pd.DataFrame,
    q: float = 0.2,
    min_assets: int = 5
) -> pd.DataFrame:
    """
    Same as make_long_short_weights, but only ranks among assets marked tradable=True at each time.
    Useful for a liquidity filter (e.g., top 50% volume rank per bar).
    """
    w = pd.DataFrame(0.0, index=score.index, columns=score.columns)

    # Ensure alignment
    tradable = tradable.reindex(index=score.index, columns=score.columns).fillna(False)

    for t in score.index:
        s = score.loc[t]

        # keep only tradable assets at time t
        s = s[tradable.loc[t]]
        s = s.dropna()
        n = len(s)
        if n < min_assets:
            continue

        k = max(1, int(n * q))
        longs = s.sort_values(ascending=False).head(k).index
        shorts = s.sort_values(ascending=True).head(k).index

        w.loc[t, longs] =  1.0 / k
        w.loc[t, shorts] = -1.0 / k

    return w


def scale_to_gross_one(w: pd.DataFrame, target_gross: float = 1.0) -> pd.DataFrame:
    """
    Scale each row so sum(abs(w_i)) == target_gross.
    If a row has gross 0, it stays 0.
    """
    gross = w.abs().sum(axis=1)
    scaled = w.div(gross.replace(0.0, np.nan), axis=0).fillna(0.0)
    return scaled * target_gross


# -----------------------------
# Backtest
# -----------------------------
def backtest_unconstrained(
    rets: pd.DataFrame,
    w: pd.DataFrame,
    cost_bps: float = 20.0,
    rebalance_every: int = 6
) -> pd.DataFrame:
    """
    Backtest with:
      - Rebalance every N bars (default 6 = daily on 4H bars)
      - Weight lag of 1 bar to avoid look-ahead
      - Turnover costs: cost = turnover * (cost_bps/10000)

    Returns DataFrame with: gross_ret, turnover, cost, net_ret, equity
    """
    # Align weights to returns
    w = w.reindex(index=rets.index, columns=rets.columns).fillna(0.0)

    # Only allow weights to change on rebalance rows; carry forward otherwise.
    mask = (np.arange(len(w)) % rebalance_every) == 0
    w = w.copy()
    w.loc[~mask, :] = np.nan
    w = w.ffill().fillna(0.0)

    # Lag weights by 1 bar (hold w_{t-1} during return at t)
    w_lag = w.shift(1).fillna(0.0)

    gross_ret = (w_lag * rets).sum(axis=1)
    turnover = (w - w_lag).abs().sum(axis=1)
    cost = turnover * (cost_bps / 10000.0)
    net_ret = gross_ret - cost

    out = pd.DataFrame(
        {"gross_ret": gross_ret, "turnover": turnover, "cost": cost, "net_ret": net_ret},
        index=rets.index
    )
    out["equity"] = (1.0 + out["net_ret"]).cumprod()
    return out


# -----------------------------
# Performance stats
# -----------------------------
def perf_stats(ret: pd.Series, turnover: pd.Series, bars_per_year: int = 365 * 6) -> Dict[str, float]:
    """
    Basic stats on returns series.
    bars_per_year: for 4H bars -> 6 per day -> 365*6.
    """
    ret = ret.dropna()
    if len(ret) == 0:
        return {"CAGR": np.nan, "Vol": np.nan, "Sharpe": np.nan, "MaxDD": np.nan, "AvgTurnover": np.nan}

    mean = ret.mean()
    vol = ret.std(ddof=1)

    sharpe = np.nan
    if vol > 0:
        sharpe = (mean / vol) * np.sqrt(bars_per_year)

    equity = (1.0 + ret).cumprod()
    dd = equity / equity.cummax() - 1.0
    max_dd = float(dd.min())

    cagr = float(equity.iloc[-1] ** (bars_per_year / len(ret)) - 1.0)

    return {
        "CAGR": cagr,
        "Vol": float(vol * np.sqrt(bars_per_year)),
        "Sharpe": float(sharpe),
        "MaxDD": max_dd,
        "AvgTurnover": float(turnover.mean()),
    }


# -----------------------------
# Signals
# -----------------------------
def momentum_score(close_px: pd.DataFrame, L: int) -> pd.DataFrame:
    """Momentum: close/close.shift(L) - 1"""
    return close_px / close_px.shift(L) - 1.0


def reversal_score(rets: pd.DataFrame, H: int) -> pd.DataFrame:
    """Reversal: negative of recent H-bar return"""
    return -(rets.rolling(H).sum())


# -----------------------------
# Sweeps / diagnostics
# -----------------------------
@dataclass
class SweepResult:
    table: pd.DataFrame
    bt_by_param: Dict[int, pd.DataFrame]


def sweep_momentum(
    rets: pd.DataFrame,
    close_px: pd.DataFrame,
    Ls: Iterable[int],
    q: float = 0.2,
    cost_bps: float = 20.0,
    rebalance_every: int = 6,
    target_gross: float = 1.0,
) -> SweepResult:
    rows: List[Dict[str, float]] = []
    bt_by_L: Dict[int, pd.DataFrame] = {}

    px = close_px.reindex(index=rets.index, columns=rets.columns)

    for L in Ls:
        score = momentum_score(px, L).reindex(rets.index)

        w = make_long_short_weights(score, q=q)
        w = scale_to_gross_one(w, target_gross=target_gross)

        bt = backtest_unconstrained(rets, w, cost_bps=cost_bps, rebalance_every=rebalance_every)
        s = perf_stats(bt["net_ret"], bt["turnover"])

        s["L"] = int(L)
        s["FinalEquity"] = float(bt["equity"].iloc[-1])
        rows.append(s)
        bt_by_L[int(L)] = bt

    tbl = pd.DataFrame(rows).sort_values("Sharpe", ascending=False).reset_index(drop=True)
    return SweepResult(table=tbl, bt_by_param=bt_by_L)


def sweep_reversal(
    rets: pd.DataFrame,
    Hs: Iterable[int],
    q: float = 0.2,
    cost_bps: float = 20.0,
    rebalance_every: int = 6,
    target_gross: float = 1.0,
    tradable: Optional[pd.DataFrame] = None,
) -> SweepResult:
    rows: List[Dict[str, float]] = []
    bt_by_H: Dict[int, pd.DataFrame] = {}

    for H in Hs:
        score = reversal_score(rets, H)

        if tradable is None:
            w = make_long_short_weights(score, q=q)
        else:
            w = make_long_short_weights_masked(score, tradable=tradable, q=q)

        w = scale_to_gross_one(w, target_gross=target_gross)

        bt = backtest_unconstrained(rets, w, cost_bps=cost_bps, rebalance_every=rebalance_every)
        s = perf_stats(bt["net_ret"], bt["turnover"])

        s["H"] = int(H)
        s["FinalEquity"] = float(bt["equity"].iloc[-1])
        rows.append(s)
        bt_by_H[int(H)] = bt

    tbl = pd.DataFrame(rows).sort_values("Sharpe", ascending=False).reset_index(drop=True)
    return SweepResult(table=tbl, bt_by_param=bt_by_H)


def pick_best_param(tbl: pd.DataFrame, param_col: str, metric: str = "Sharpe") -> int:
    """Pick best parameter from a sweep table by metric (default Sharpe)."""
    return int(tbl.sort_values(metric, ascending=False).iloc[0][param_col])


def stress_test_costs(
    rets: pd.DataFrame,
    w: pd.DataFrame,
    costs_bps: Iterable[float] = (20.0, 40.0, 60.0),
    rebalance_every: int = 6
) -> pd.DataFrame:
    rows = []
    for c in costs_bps:
        bt = backtest_unconstrained(rets, w, cost_bps=float(c), rebalance_every=rebalance_every)
        s = perf_stats(bt["net_ret"], bt["turnover"])
        s["cost_bps"] = float(c)
        s["FinalEquity"] = float(bt["equity"].iloc[-1])
        rows.append(s)
    return pd.DataFrame(rows).sort_values("cost_bps").reset_index(drop=True)


def train_test_split_stats(bt: pd.DataFrame, cut: str) -> pd.DataFrame:
    """Return a 2-row table of stats for Train (<cut) and Test (>=cut)."""
    train = bt.loc[bt.index < cut]
    test = bt.loc[bt.index >= cut]
    train_stats = perf_stats(train["net_ret"], train["turnover"])
    test_stats = perf_stats(test["net_ret"], test["turnover"])
    out = pd.DataFrame([train_stats, test_stats], index=[f"Train (< {cut})", f"Test (>= {cut})"])
    return out[["Sharpe", "CAGR", "Vol", "MaxDD", "AvgTurnover"]]


def bucket_diagnostic(
    rets: pd.DataFrame,
    score: pd.DataFrame,
    n_buckets: int = 5
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Sort assets into n_buckets by score each bar, and compute next-bar return by bucket.
    Returns:
      - bucket_means: mean next-bar return per bucket (Series)
      - bucket_df: time series of bucket returns (DataFrame)
    """
    signal = score.dropna(how="all")
    future_ret = rets.shift(-1).reindex(signal.index)

    # rank within each row into [0,1]
    q_rank = signal.rank(axis=1, pct=True)

    # map to bucket ids 0..n_buckets-1
    bucket = (q_rank * n_buckets).astype(int).clip(0, n_buckets - 1)

    bucket_ts = {}
    for b in range(n_buckets):
        mask = bucket == b
        bucket_ts[b] = future_ret.where(mask).mean(axis=1)

    bucket_df = pd.DataFrame(bucket_ts)
    bucket_means = bucket_df.mean()
    return bucket_means, bucket_df
