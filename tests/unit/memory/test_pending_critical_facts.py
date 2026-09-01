"""In-memory pending critical-fact store (D34). No Postgres required."""
from backend.memory.pending_critical_facts import (
    InMemoryPendingCriticalFactStore,
    PendingCriticalFact,
)


def test_put_get_delete_round_trip():
    store = InMemoryPendingCriticalFactStore()
    fact = PendingCriticalFact(
        pending_id="p1",
        user_id="default_user",
        session_id="s1",
        allergens=("甲壳类",),
        supplements=("鱼油",),
    )
    store.put(fact)
    loaded = store.get("p1")
    assert loaded is not None
    assert loaded.allergens == ("甲壳类",)
    assert loaded.supplements == ("鱼油",)
    assert store.list_for_session("s1")[0].pending_id == "p1"
    assert store.delete("p1") is not None
    assert store.get("p1") is None


def test_event_dict_does_not_claim_already_recorded():
    fact = PendingCriticalFact(
        pending_id="p1",
        user_id="u",
        session_id="s",
        allergens=("甲壳类",),
    )
    event = fact.to_event_dict()
    assert event["pending_id"] == "p1"
    assert "甲壳类" in event["detail"]
    assert "已记录" not in event["detail"]
