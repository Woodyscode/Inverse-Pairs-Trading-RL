"""Write the decision funnel and cross-asset coverage of an inverse-pair run."""

import numpy as np
import pandas as pd


def write_report(output, frames):
    screen = frames["candidate_screen.csv.gz"]
    test = frames["test_decisions.csv"]
    funds = frames["etf_coverage.csv"]
    passed = screen.status.eq("screened")
    rho = screen.rho.le(-0.8) & screen.status.ne("history")
    groups = ["category", "asset_mix"]
    labels = np.sort(screen[["asset_a", "asset_b"]].to_numpy(), axis=1)
    mix = (
        pd.Series(labels[:, 0], index=screen.index)
        + " / "
        + pd.Series(labels[:, 1], index=screen.index)
    )
    counts = (
        screen.assign(asset_mix=mix, strict=rho, liquid=passed)
        .groupby(groups)
        .agg(
            pair_windows=("a", "size"),
            strict=("strict", "sum"),
            strict_and_liquid=("liquid", "sum"),
        )
    )
    counts.to_csv(output / "asset_class_funnel.csv")
    overview = counts.groupby("category").sum()
    leaders = test.status.ne("cash")
    overview["selected_windows"] = (
        test[leaders].groupby("category").size().reindex(overview.index, fill_value=0)
    )
    overview["test_net_pct"] = test.groupby("category").net.apply(
        lambda s: 100 * ((1 + s).prod() - 1) if s.notna().all() else np.nan
    )
    table = overview.reset_index().to_markdown(index=False, floatfmt=".4f")
    composition = funds.groupby("asset_class").observations.agg(
        configured="size", with_prices=lambda x: int(x.gt(0).sum())
    )
    chosen = (
        test.loc[
            leaders,
            ["window", "category", "a", "b", "mode", "horizon", "status", "net"],
        ]
        if leaders.any()
        else None
    )
    chosen_table = (
        "Every group held cash throughout."
        if chosen is None
        else chosen.to_markdown(index=False, floatfmt=".6f")
    )
    report = f"""# Strict inverse-pair discovery

The rolling study began with {len(screen):,} pair-window candidates. {int(rho.sum()):,} had complete training history and return correlation at or below −0.80. {int(passed.sum()):,} also met the $10 million per-leg liquidity floor. Counts include repeated observations of the same pair across folds. Each fold simulates a $100,000 account per category. The test return below compounds those fold returns and includes the cash windows.

{table}

## Universe and selection

{composition.to_markdown()}

The ETF list is a fixed pre-2014 cohort from contemporaneous filings, combined with stocks that belonged to the S&P 500 at each training cutoff. Historical data are reconstructed. The ETF list covers a subset of funds available at the time. Missing histories remain in the screening ledger. Sector ETFs and stocks are screened against hedge instruments. The cross-asset ETF category requires different asset classes. Distinct factors within one macro class, such as euro versus dollar, have a separate category. The inverse ETF category pairs opposite exposures to a common benchmark or proxy.

## Validation decisions

Each training window fits a daily return hedge. Its tracking-error signal measures the latest departure from that relationship. A second signal uses log-price spread deviations and requires a negative log-price hedge and a Holm-adjusted Engle–Granger p-value below 0.05. Both signals use the previous 30 sessions to normalize today's deviation. The return correlation must also remain at or below −0.80 in all three training subwindows and the separate validation window. The log-price test gates the stationary-spread hypothesis alone.

A signal of at least one standard deviation in magnitude requests a 25% gross position at the next close. Fixed entry units are held for one, five or ten sessions. Negative beta gives weights proportional to (1, −beta), so a long spread buys both legs. The validation ledger records normalized leg weights, estimated market beta and account capacity separately. Same-side positions can carry outright market risk. Designed daily inverse targets and futures roll effects can generate tracking relationships and compounded drift without a tradable price discrepancy.

Each simulated fill incurs two basis points on each leg's traded notional. Financing is 2% annually on gross positions and borrow is 0.5% annually on shorts, accrued by calendar day. Both legs must fit within 1% of trailing and execution-day dollar volume. An unfillable exit invalidates the candidate simulation. Historical quotes, stock-loan locates and time-varying borrow fees are unavailable. These are modeled close-price fills.

Each qualifying rule needs eight completed validation trades, positive net validation return, and a positive predictive slope significant at 5% after Holm correction across eligible pair, signal and horizon tests within the fold. Slope tests use overlapping forward basket returns with autocorrelation-robust standard errors. Realized trades never overlap within a candidate. Validation net return ranks qualifying rules within category, with validation correlation breaking ties. The top rule is frozen before the next test window. The six-month windows and repeated discovery across folds limit statistical confidence.

## Frozen choices and held-out results

{chosen_table}

Net values above are fractions of starting capital for each fold. Test results never enter the rankings. The discovery run trained zero forecasting or reinforcement-learning models.

The [asset-class funnel](asset_class_funnel.csv) counts every asset combination. The [candidate screen](candidate_screen.csv.gz) records the earliest eligibility failure. The [training diagnostics](training_diagnostics.csv) contain correlation stability, designed tracking error and hedge estimates. The [validation rankings](validation_rankings.csv) list rule-specific rejection reasons and ranks. The [test decisions](test_decisions.csv), [trade ledger](test_trades.csv), [ETF coverage](etf_coverage.csv) and [manifest](manifest.json) support reproduction.
"""
    (output / "report.md").write_text(report, encoding="utf-8")
