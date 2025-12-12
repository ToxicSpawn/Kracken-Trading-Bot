from datetime import datetime

from core.policy import PolicyConfig, PolicyEngine
from core.state import AccountState, GlobalState, PositionSnapshot


def test_policy_engine_flags_multiple_violations():
    cfg = PolicyConfig(
        max_daily_loss_aud=-100.0,
        absolute_max_drawdown_pct=-10.0,
        max_single_position_pct=0.4,
        per_symbol_max_loss_aud=-50.0,
        allowed_trading_hours_utc=None,
    )
    engine = PolicyEngine(cfg)

    pos = PositionSnapshot(symbol="BTC/AUD", realized_pnl=-75.0)
    acct = AccountState(name="primary", equity=900.0, balance=900.0, positions={"BTC/AUD": pos})
    state = GlobalState(accounts={"primary": acct}, total_realized_pnl=-150.0)

    decision = engine.evaluate(state)

    assert not decision.can_trade
    assert any("daily PnL" in r for r in decision.reasons)
    assert any("BTC/AUD" in r for r in decision.reasons)


def test_policy_engine_drawdown_detection():
    cfg = PolicyConfig(
        max_daily_loss_aud=-500.0,
        absolute_max_drawdown_pct=-5.0,
        max_single_position_pct=0.4,
        per_symbol_max_loss_aud=-500.0,
        allowed_trading_hours_utc=None,
    )
    engine = PolicyEngine(cfg)

    acct = AccountState(name="primary", equity=1000.0, balance=1000.0, positions={})
    state = GlobalState(accounts={"primary": acct}, total_realized_pnl=0.0)
    # Prime equity peak
    assert engine.evaluate(state).can_trade

    # Drawdown beyond threshold
    acct.equity = 900.0
    decision = engine.evaluate(state)

    assert not decision.can_trade
    assert any("drawdown" in r for r in decision.reasons)


def test_policy_engine_trading_hours_gate(monkeypatch):
    # Ensure disallowed hour by picking the next hour modulo 24
    next_hour = (datetime.utcnow().hour + 1) % 24
    cfg = PolicyConfig(
        max_daily_loss_aud=-500.0,
        absolute_max_drawdown_pct=-50.0,
        max_single_position_pct=0.4,
        per_symbol_max_loss_aud=-500.0,
        allowed_trading_hours_utc=[next_hour],
    )
    engine = PolicyEngine(cfg)

    acct = AccountState(name="primary", equity=1000.0, balance=1000.0, positions={})
    state = GlobalState(accounts={"primary": acct}, total_realized_pnl=0.0)

    decision = engine.evaluate(state)

    assert not decision.can_trade
    assert any("trading hour" in r for r in decision.reasons)
