from analytics.scorecard import BotScorecard, compute_scorecard
from core.state import AccountState, GlobalState


def test_scorecard_rewards_clean_state() -> None:
    state = GlobalState()
    card = compute_scorecard(state)

    assert isinstance(card, BotScorecard)
    assert card.score == 20.0
    assert card.rating == "20/10"
    assert card.reasons == ["All systems nominal"]


def test_scorecard_penalizes_risks_and_pauses() -> None:
    state = GlobalState(
        total_realized_pnl=-50.0,
        total_unrealized_pnl=-10.0,
        trading_enabled=False,
        meta={
            "data_quality": {"XBTUSDT": {"5m": {"gap": True, "stale": False}}},
            "policy_reasons": ["drawdown_limit"],
        },
        accounts={
            "paper": AccountState(name="paper", risk_multiplier=2.0),
        },
    )

    card = compute_scorecard(state)

    assert card.score < 10.0
    assert card.rating == "needs_attention"
    assert "Trading paused" in card.reasons[0]
    assert any("Negative PnL" in reason for reason in card.reasons)
    assert any("Data-quality" in reason for reason in card.reasons)
    assert any("Risk multiplier" in reason for reason in card.reasons)
