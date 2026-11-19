import numpy as np
import pandas as pd
import pandas as pd

from backtest.walkforward import WalkForwardEngine
from data.loaders import DataLoader
from data.returns import compute_log_returns
from env.features.feature_selector import FeatureSelector
from utils.pairs_selection import find_strong_pairs


def _config():
    return {
        "data": {"price_type": "Adj Close"},
        "pairs": {
            "correlation_threshold": 0.6,
            "allow_positive": False,
            "allow_negative": True,
            "use_cointegration": True,
            "cointegration_p_value": 0.10,
            "min_observations": 80,
            "traditional_min_annual_vol": None,
            "traditional_max_annual_vol": None,
            "inverse_min_annual_vol": None,
            "inverse_max_annual_vol": None,
            "max_traditional_pairs": 0,
            "max_inverse_pairs": None,
        },
        "env": {"window_size": 30, "initial_capital": 100000},
        "walkforward": {
            "train_years": 1,
            "test_months": 3,
            "min_train_size": 200,
            "min_test_size": 40,
            "mode": "rolling",
            "feature_context_days": 252,
        },
        "training": {},
        "features": {"min_periods": 60, "base_window": 30},
    }


def _synthetic_panels(periods=390):
    return DataLoader._synthetic_ohlcv(
        tickers=["SMOKE_A", "SMOKE_B", "SMOKE_C"],
        start="2020-01-01",
        periods=periods,
        seed=42,
    )


def _aligned_returns(ohlcv):
    returns = compute_log_returns(ohlcv["Adj Close"])
    idx = returns.index
    return {k: v.loc[idx].copy() for k, v in ohlcv.items()}, returns


def test_train_validation_and_test_features_are_nonzero_with_context():
    ohlcv, returns = _aligned_returns(_synthetic_panels(390))
    engine = WalkForwardEngine(
        ohlcv=ohlcv,
        returns=returns,
        config=_config(),
        env_builder=lambda **kwargs: None,
        price_index=returns.index,
    )
    pair = ("SMOKE_A", "SMOKE_B")
    train = {k: v.iloc[:252].copy() for k, v in ohlcv.items()}
    validation = {k: v.iloc[252:315].copy() for k, v in ohlcv.items()}
    test = {k: v.iloc[315:378].copy() for k, v in ohlcv.items()}
    context = {k: v.iloc[:252].copy() for k, v in ohlcv.items()}
    val_context = {k: v.iloc[63:315].copy() for k, v in ohlcv.items()}

    train_features, _ = engine._compute_features_for_window(train, [pair])
    val_features, _ = engine._compute_features_for_window(validation, [pair], context_ohlcv=context)
    test_features, _ = engine._compute_features_for_window(test, [pair], context_ohlcv=val_context)

    assert np.any(train_features[pair] != 0.0)
    assert np.any(val_features[pair] != 0.0)
    assert np.any(test_features[pair][0] != 0.0)
    changed = {key: frame.iloc[:1].copy() for key, frame in test.items()}
    prefix, _ = engine._compute_features_for_window(changed, [pair], context_ohlcv=val_context)
    np.testing.assert_allclose(prefix[pair][0], test_features[pair][0])


def _select_first_window_pairs(ohlcv):
    ohlcv, returns = _aligned_returns(ohlcv)
    engine = WalkForwardEngine(
        ohlcv=ohlcv,
        returns=returns,
        config=_config(),
        env_builder=lambda **kwargs: None,
        price_index=returns.index,
    )
    ohl_train, _, _, _, _, _ = next(engine._generate_windows())
    prices = ohl_train["Adj Close"]
    rets = prices.pct_change().dropna()
    return find_strong_pairs(rets, prices, _config())


def test_pair_selection_uses_training_window_not_test_window():
    base = _synthetic_panels(340)
    selected_base = _select_first_window_pairs(base)

    test_mutated = {k: v.copy() for k, v in base.items()}
    for panel in test_mutated.values():
        panel.iloc[260:] = panel.iloc[260:].iloc[::-1].values

    selected_test_mutated = _select_first_window_pairs(test_mutated)
    assert selected_test_mutated == selected_base

    train_mutated = {k: v.copy() for k, v in base.items()}
    for field in train_mutated:
        train_mutated[field].loc[:, "SMOKE_B"] = train_mutated[field]["SMOKE_C"].values

    selected_train_mutated = _select_first_window_pairs(train_mutated)
    assert selected_train_mutated != selected_base


def test_feature_scaler_is_fit_on_train_only():
    rng = np.random.default_rng(123)
    x_train = rng.normal(0, 1, size=(120, 4))
    y_train = x_train[:, 0] * 0.5
    x_test = rng.normal(100, 1, size=(20, 4))

    selector = FeatureSelector(alpha=0.001, l1_ratio=0.5, max_iter=5000, auto_k=False)
    selector.fit_pair(("A", "B"), x_train, y_train)
    transformed = selector.transform_pair(("A", "B"), x_test)

    selected = selector.selected_features[("A", "B")]
    expected = selector.scalers[("A", "B")].transform(x_test)[:, selected]
    np.testing.assert_allclose(transformed, expected)
    assert not np.allclose(np.mean(transformed, axis=0), 0.0)


def test_validation_windows_are_ordered_and_non_overlapping():
    ohlcv, returns = _aligned_returns(_synthetic_panels(320))
    cfg = _config()
    cfg["walkforward"]["train_years"] = 0.75
    cfg["walkforward"]["validation_months"] = 2
    cfg["walkforward"]["test_months"] = 2
    cfg["walkforward"]["min_train_size"] = 150
    cfg["walkforward"]["min_validation_size"] = 30
    cfg["walkforward"]["step_days"] = 42

    engine = WalkForwardEngine(
        ohlcv=ohlcv,
        returns=returns,
        config=cfg,
        env_builder=lambda **kwargs: None,
        price_index=returns.index,
    )

    ohl_train, ohl_val, ohl_test, *_ = next(engine._generate_validation_windows())

    assert ohl_train["Close"].index.max() < ohl_val["Close"].index.min()
    assert ohl_val["Close"].index.max() < ohl_test["Close"].index.min()

def test_training_hedge_target_matches_next_bar_basket_return():
    from env.features.targets import build_targets
    from utils.pairs_selection import select_pair_models
    prices = pd.DataFrame({"A": [100., 102., 98., 104.], "B": [50., 49., 51., 48.]})
    config = _config()
    config["pairs"].update(use_cointegration=False, min_observations=3)
    model, = select_pair_models(prices.pct_change(), prices, config)
    assert np.isclose(model["hedge_ratio"], -2)
    pair = ("A", "B")
    revised = pd.concat([prices, pd.DataFrame({"A": [108.], "B": [49.]})], ignore_index=True)
    targets = build_targets({"Adj Close": revised}, [pair], pair_models={pair: model})
    assert np.isclose(targets[pair][-2], (4 + 2 * 1) / (104 + 2 * 48))
    assert np.isnan(targets[pair][-1])


def test_bar_metrics_include_month_boundary_and_pairwise_benchmark_dates():
    from backtest.metrics import compute_all_metrics
    index = pd.to_datetime(["2024-01-30", "2024-01-31", "2024-02-01", "2024-02-02"])
    curve = pd.Series([100., 99., 110., 108.9], index=index)
    spy = pd.Series([100., 99., np.nan, 101.], index=index)
    metrics = compute_all_metrics(curve, spy, curve.diff().dropna())
    assert np.isclose(metrics["monthly_returns"][pd.Timestamp("2024-02-29")], 0.10)
    assert np.isclose(metrics["bar_win_rate"], 1 / 3)

def test_holm_adjusts_across_all_screened_pairs_before_truncation(monkeypatch):
    from utils.pairs_selection import select_pair_models
    prices = pd.DataFrame({"A": [10., 12, 11, 13], "B": [20., 18, 19, 17], "C": [40., 36, 38, 34]})
    config = _config()
    config["pairs"].update(min_observations=3, cointegration_p_value=0.05, multiple_testing="holm")
    monkeypatch.setattr("utils.pairs_selection.coint", lambda *args: (0, 0.04, None))
    assert select_pair_models(prices.pct_change(), prices, config) == []
    config["pairs"]["multiple_testing"] = "none"
    assert len(select_pair_models(prices.pct_change(), prices, config)) == 2


def test_no_pair_windows_preserve_the_test_calendar():
    ohlcv, returns = _aligned_returns(_synthetic_panels(370))
    cfg = _config()
    cfg["pairs"].update(allow_positive=False, allow_negative=False)
    cfg["walkforward"].update(train_years=0.75, min_train_size=150,
                              validation_months=2, min_validation_size=30, test_months=2)
    cfg["benchmark_curve"] = pd.Series(1.0, index=returns.index)
    engine = WalkForwardEngine(ohlcv, returns, cfg, None, returns.index)
    windows = list(engine._generate_validation_windows())
    result = engine.run()
    expected = windows[0][2]["Close"].index.append([w[2]["Close"].index for w in windows[1:]])
    assert result["equity_curve"].index.equals(expected)
    assert (result["equity_curve"] == cfg["env"]["initial_capital"]).all()
    assert all(w["models"] == [] for w in result["selected_pairs"])
