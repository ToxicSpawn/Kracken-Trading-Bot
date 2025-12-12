from strategies.base import SignalType
from strategies.rsi_ml_strategy import RsiMlStrategy
from strategies.trend_vol_strategy import TrendVolStrategy


def build_ohlcv(closes):
    return [[i, price, price, price, price, 1.0] for i, price in enumerate(closes)]


def test_rsi_strategy_thresholds_and_confidence():
    strat = RsiMlStrategy("TEST")

    buy_data = build_ohlcv(list(range(140, 100, -1)))
    buy_signal = strat.generate_signal(buy_data)
    assert buy_signal.signal is SignalType.BUY
    assert 0.3 <= buy_signal.confidence <= 0.95

    sell_data = build_ohlcv(list(range(50, 90)) + [92, 93, 94, 93, 92, 90, 88, 87])
    sell_signal = strat.generate_signal(sell_data)
    assert sell_signal.signal is SignalType.SELL
    assert 0.2 <= sell_signal.confidence <= 0.95

    hold_data = build_ohlcv([10] * 35)
    hold_signal = strat.generate_signal(hold_data)
    assert hold_signal.signal is SignalType.HOLD
    assert 0.19 <= hold_signal.confidence <= 0.5


def test_trend_vol_scores_with_small_and_flat_series():
    strat = TrendVolStrategy("TEST")

    short_history = build_ohlcv([1, 2, 3, 4, 5])
    trend_score, vol_score = strat.score(short_history)
    assert trend_score == 0.0
    assert vol_score == 0.0

    flat_history = build_ohlcv([10] * 35)
    trend_score, vol_score = strat.score(flat_history)
    assert 0.49 <= trend_score <= 0.51
    assert vol_score == 0.0


def test_trend_vol_scores_clip_and_bounds():
    strat = TrendVolStrategy("TEST")
    trending_prices = list(range(1, 61))
    trend_score, vol_score = strat.score(build_ohlcv(trending_prices))

    assert trend_score > 0.8
    assert 0.0 <= vol_score <= 1.0
