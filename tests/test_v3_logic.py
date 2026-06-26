"""v3 logic tests — pure functions, no models/faiss/services required.

    python -m pytest tests/test_v3_logic.py      (or just run this file)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.scoring.composite import score_match, lane_for
from src.llm_verifier import decide, ALLOW, REVIEW
import src.llm_verifier as lv


def test_metadata_does_not_rescue_below_gate():
    r = score_match(0.50, 1.0, False, 0.9)
    assert r["duplicate_type"] is None
    assert r["metadata_score"] is None        # never computed below the gate
    assert r["match_confidence"] == 0.50       # no lift applied


def test_same_asset_vs_re_authored():
    assert score_match(0.75, 1.0, False, 0.90)["duplicate_type"] == "same_asset"
    assert score_match(0.75, 1.0, False, 0.40)["duplicate_type"] == "re_authored"


def test_cross_medium_has_no_feature_score():
    r = score_match(0.75, 1.0, True, None)
    assert r["duplicate_type"] == "cross_medium" and r["feature_score"] is None


def test_metadata_lift_is_capped():
    # full metadata on a 0.60 text match cannot reach the 0.85 urgent band
    assert score_match(0.60, 1.0, True, None)["match_confidence"] == 0.75


def test_lane_routing():
    assert lane_for("same_asset") == "fast"
    assert lane_for("cross_medium") == "investigation"
    assert lane_for("re_authored") == "investigation"


def test_no_auto_removal():
    assert not hasattr(lv, "REMOVE"), "v3 must not define an automated REMOVE"
    assert decide(0.20)["decision"] == ALLOW
    assert decide(0.45)["decision"] == REVIEW
    # high confidence + strong LLM still only escalates priority, never removes
    lv._call_ollama = lambda prompt: (0.95, "stub")
    d = decide(0.90)
    assert d["decision"] == REVIEW and d["priority"] == "urgent"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"PASS {name}")
    print("\nall v3 logic tests passed")
