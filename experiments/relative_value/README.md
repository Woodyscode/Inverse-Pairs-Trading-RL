# Daily and one-minute relative-value research

The follow-up tests whether intraday information improves a daily Ridge decision at holding periods of one, five or ten sessions. The matched daily and minute models use identical training and evaluation dates, target returns, execution prices, capital, hedge weights, costs and liquidity rules. The minute model adds morning residual movement and realized residual volatility. These use synchronized positive-volume one-minute bars from 09:30 through 12:59 New York time, with at least 200 of 209 possible one-minute returns per morning. The fixed interval also fits early-close sessions. Gaps and overnight jumps are excluded. Both models place orders at the following daily close.

The price panel uses reconstructed historical S&P 500 membership intervals, distinct identifiers for former constituents and the source's nontrading-day flags. It is a later reconstruction of the historical record. Each of the 13 original folds had 503–505 constituents and 488–498 stocks with complete training data and at least $10 million median daily dollar volume over the final 60 training sessions. The market-hedged strategy selected the most liquid stock and hedged it against SPY using its trailing 120-session return beta. The separate cointegration strategy tested all 435 combinations of the 30 most liquid names per fold, with a fixed regression direction, one lag and a within-fold Holm correction at 5%. Five pair-window tests passed across two folds. The implementation selected one pair per qualifying fold and estimated its hedge from trailing log prices. [Historical membership and price source](https://github.com/Johnbrick123/sp500-data) · [Minute-bar source and revision](https://huggingface.co/datasets/syncster-data/historical-market-data/tree/eff0dac55d843b3b83caec819efa213021a3e3e5) · [Fold selections](../../outputs/runs/rv_matched_minute/folds.csv)

Training labels end by the training cutoff. Each validation or test forecast earns the basket return from the following close through its specified exit close. Forecast scores include overlapping labels. The trading ledger opens nonoverlapping positions on a fixed schedule, holds the entry quantities, charges both fills plus daily financing and short borrow, and caps each fill at 1% of the lower of trailing and execution-day dollar volume. Each portfolio starts at $100,000, targets 25% gross exposure, and holds cash between trades or when forecasts are unavailable. Borrow uses a fixed rate with assumed availability. The ledger models close-price fills.

| Construction | Holding period | Matched daily inputs | Daily plus minute inputs | Difference |
| --- | ---: | ---: | ---: | ---: |
| Stock hedged with SPY | 1 session | −4.59% | +3.72% | +8.32 pp |
| Stock hedged with SPY | 5 sessions | −2.54% | −1.50% | +1.04 pp |
| Stock hedged with SPY | 10 sessions | +3.01% | −4.63% | −7.64 pp |
| Cointegrated stocks | 1 session | −3.80% | −3.64% | +0.16 pp |
| Cointegrated stocks | 5 sessions | −8.38% | −7.86% | +0.52 pp |
| Cointegrated stocks | 10 sessions | −5.22% | −0.90% | +4.32 pp |

The table shows cumulative held-out net returns from July 2017 through December 2023, including cash periods. The one-session market-hedged comparison scored 1,485 paired forecasts, and the cointegrated comparison scored 214. None of the six differences in mean daily returns passed a two-sided 5% test using 20-lag HAC standard errors and Holm adjustment. The smallest adjusted p-value was 0.601. A separate full-daily-history control evaluates the opportunity cost of requiring minute coverage. Each minute comparison must use its matched daily control. ULE/UUP had zero qualifying mornings across its two original test windows, so the feed cannot measure this effect for the original inverse pair. [Paired tests](../../outputs/runs/rv_matched_minute/granularity.csv) · [All strategy results](../../outputs/runs/rv_matched_minute/summary.csv) · [Original inverse-pair coverage](../../outputs/runs/rv_matched_minute/inverse_minute_coverage.csv)

Run the commands from the repository root using a new output directory. The restoration command verifies each cached input by SHA-256 and can download missing inputs. The daily provider updates its release in place, so retain the frozen cache for exact historical reproduction. The Parquet files are in the local data cache at `outputs/runs/cache/rv_sp500/`.

```bash
python -m data.fetch_relative_value
python -m experiments.relative_value.relative_value_horizons --minute --output outputs/runs/my_granularity_study
```

