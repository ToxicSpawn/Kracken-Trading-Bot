from core.intent_netting import net_intents, NettingConfig


def test_net_intents_basic():
    intents = [
        {"account": "a", "symbol": "BTC/AUD", "side": "BUY", "qty": 0.3, "type": "market"},
        {"account": "a", "symbol": "BTC/AUD", "side": "BUY", "qty": 0.2, "type": "market"},
        {"account": "a", "symbol": "BTC/AUD", "side": "SELL", "qty": 0.1, "type": "market"},
    ]
    out = net_intents(intents, NettingConfig(keep_sides_separate=False))
    assert len(out) == 1
    assert out[0]["side"] == "BUY"
    assert abs(out[0]["qty"] - 0.4) < 1e-9


def test_net_intents_keep_sides():
    intents = [
        {"account": "a", "symbol": "ETH/AUD", "side": "BUY", "qty": 1.0, "type": "market"},
        {"account": "a", "symbol": "ETH/AUD", "side": "SELL", "qty": 0.4, "type": "market"},
    ]
    out = net_intents(intents, NettingConfig(keep_sides_separate=True))
    assert len(out) == 2

