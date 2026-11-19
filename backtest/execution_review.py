"""Reconcile published ledgers and reproduce the execution research exhibits offline."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUN = Path("outputs/runs/strict_inverse_alpha_final")
BOOK = ["horizon", "model"]


def reconcile(folder):
    trades = pd.read_parquet(folder / "trades.parquet")
    fills = pd.read_parquet(folder / "executions.parquet")
    equity = pd.read_parquet(folder / "daily_equity.parquet")
    metrics = pd.read_csv(folder / "metrics.csv").set_index(BOOK).sort_index()
    np.testing.assert_allclose(fills.units * fills.price, fills.notional, atol=1e-7)
    np.testing.assert_allclose(fills.fee, fills.notional.abs() * 0.0002, atol=1e-7)
    np.testing.assert_allclose(
        fills.groupby(BOOK + ["entry_day", "symbol"]).units.sum(), 0, atol=1e-7
    )
    np.testing.assert_allclose(
        trades.gross - trades.execution - trades.funding - trades.borrow,
        trades.net,
        atol=1e-7,
    )
    executions = (
        fills.assign(turnover=fills.notional.abs())
        .groupby(BOOK)
        .agg(execution=("fee", "sum"), turnover=("turnover", "sum"))
    )
    ledgers = trades.groupby(BOOK)[["execution", "turnover"]].sum()
    pd.testing.assert_frame_equal(executions, ledgers, atol=1e-7, rtol=1e-11)
    daily = equity.groupby(BOOK).pnl.sum()
    pd.testing.assert_index_equal(daily.index, metrics.index)
    np.testing.assert_allclose(
        daily,
        trades.groupby(BOOK).net.sum().reindex(daily.index, fill_value=0),
        atol=1e-7,
    )
    np.testing.assert_allclose(daily, 100000 * metrics.net, atol=1e-7)
    return trades


def execution_stress(book):
    turnover = book.turnover.sum()
    before_execution = (book.gross - book.funding - book.borrow).sum()
    rates = np.array([0.0, 2.0, 5.0, 10.0, 20.0])
    return pd.DataFrame(
        {
            "cost_bp_per_filled_dollar": rates,
            "execution_usd": turnover * rates / 10000,
            "net_usd": before_execution - turnover * rates / 10000,
        }
    ), 10000 * before_execution / turnover


def plot_granularity(paired, destination):
    labels = {
        "linear": "Linear",
        "ridge": "Ridge",
        "xgboost": "XGBoost",
        "reversal": "Reversal",
        "continuation": "Continuation",
    }
    fig, ax = plt.subplots(figsize=(8.2, 3.9), layout="constrained")
    ax.axvline(0, color="#506174", linewidth=1)
    for year, offset, color, label in [
        (2025, -0.13, "#566979", "2025"),
        (2026, 0.13, "#007478", "Jan–Oct 2026"),
    ]:
        sample = (
            paired[
                (paired.fold == year) & (paired.split == "test") & (paired.horizon == 1)
            ]
            .set_index("model")
            .loc[list(labels)]
        )
        x = 100 * sample.net_difference
        y = np.arange(len(sample)) + offset
        ax.hlines(y, 100 * sample.lower, 100 * sample.upper, color=color, lw=2)
        ax.scatter(x, y, color=color, label=label, s=30, zorder=3)
    ax.set_yticks(np.arange(len(labels)), labels.values())
    ax.invert_yaxis()
    ax.set_xlabel(
        "Net account return with opening-hour features minus daily-only return (percentage points)"
    )
    ax.set_title(
        "One-session trades at matched execution prices", loc="left", weight="bold"
    )
    ax.grid(axis="x", alpha=0.16)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(loc="lower right", frameon=False)
    fig.savefig(destination, dpi=170, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--output", type=Path, help="New directory for the exhibits")
    args = parser.parse_args()
    for year in (2025, 2026):
        for split in ("validation", "test"):
            trades = reconcile(args.run / str(year) / split)
            print(f"{year} {split} reconciled")
            if year == 2026 and split == "test":
                book = trades[(trades.horizon == 1) & (trades.model == "linear_minute")]
    stress, rate = execution_stress(book)
    net = book.net.sum()
    print(stress.round(2).to_string(index=False))
    print(f"Break-even total execution charge: {rate:.2f} bp per filled dollar")
    print(
        f"Net PnL excluding the two largest gains: ${net - book.net.nlargest(2).sum():,.2f}"
    )
    if args.output:
        args.output.mkdir(parents=True, exist_ok=False)
        stress.to_csv(args.output / "cost_sensitivity.csv", index=False)
        book.groupby(["a", "b"]).agg(
            trades=("net", "size"), net_usd=("net", "sum")
        ).sort_values("net_usd", ascending=False).to_csv(
            args.output / "pair_attribution.csv"
        )
        plot_granularity(
            pd.read_csv(args.run / "evidence/paired_granularity.csv"),
            args.output / "granularity.png",
        )


if __name__ == "__main__":
    main()
