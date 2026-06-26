import os
import json
import difflib
import html
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import shap
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import REPORTS_DIR, MODELS_DIR, load_manifest
from src.persistence.prediction_logger import log_reviewer_verdict

st.set_page_config(
    page_title="Harmful Video Duplicate Detection",
    page_icon="🛡️",
    layout="wide",
)

SCORE_COLOURS = {
    "REMOVE": "#ef4444",
    "HUMAN REVIEW": "#f59e0b",
    "ALLOW": "#22c55e",
}


@st.cache_resource(show_spinner="Loading model artifacts...")
def load_model():
    model = joblib.load(os.path.join(MODELS_DIR, "harmful_detector.joblib"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
    manifest = load_manifest()
    return model, scaler, manifest


@st.cache_data(show_spinner="Loading inference results...")
def load_results(split_name="test"):
    path = os.path.join(REPORTS_DIR, f"inference_results_{split_name}.parquet")
    if not os.path.exists(path):
        return None
    return pd.read_parquet(path)


@st.cache_data(show_spinner="Loading video text...")
def load_video_texts():
    """video_id -> {title_text, ocr_text, asr_text} across all splits, so we can
    show the actual matched text for both the query video and its seed matches."""
    base = os.path.join(os.path.dirname(REPORTS_DIR), "datamart", "gold", "splits")
    frames = []
    for sp in ("train", "val", "test"):
        p = os.path.join(base, f"{sp}.parquet")
        if os.path.exists(p):
            frames.append(pd.read_parquet(p, columns=["video_id", "title_text", "ocr_text", "asr_text"]))
    if not frames:
        return {}
    allrows = pd.concat(frames, ignore_index=True).drop_duplicates("video_id")
    return allrows.set_index("video_id").to_dict("index")


def _parse_pair(pair):
    """'q.title<->c.ocr' -> ('title', 'ocr')."""
    try:
        left, right = pair.split("<->")
        return left.split(".", 1)[1], right.split(".", 1)[1]
    except Exception:
        return "title", "title"


def _txt(d, key):
    v = d.get(key) if d else None
    return v if isinstance(v, str) and v.strip() else "—"


_DIFF_WRAP = ("font-family:ui-monospace,SFMono-Regular,Menlo,monospace;"
              "white-space:pre-wrap;word-break:break-word;background:#f8fafc;"
              "border:1px solid #e2e8f0;border-radius:6px;padding:8px;font-size:0.85em;"
              "line-height:1.5")
_HL = "background:#fde68a;border-radius:2px;padding:0 1px"


def diff_pair(a, b):
    """Return (a_html, b_html) with the characters that differ highlighted, so
    the evasion (leetspeak / OCR misread / homophone) is visible at a glance."""
    a, b = a or "", b or ""
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    left, right = [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        sa, sb = html.escape(a[i1:i2]), html.escape(b[j1:j2])
        if tag == "equal":
            left.append(sa)
            right.append(sb)
        else:
            if sa:
                left.append(f"<span style='{_HL}'>{sa}</span>")
            if sb:
                right.append(f"<span style='{_HL}'>{sb}</span>")
    return (f"<div style='{_DIFF_WRAP}'>{''.join(left)}</div>",
            f"<div style='{_DIFF_WRAP}'>{''.join(right)}</div>")


def score_bar(label, value, weight=None):
    wtxt = f" (weight {weight:.0%})" if weight is not None else ""
    if value is None or pd.isna(value):
        st.markdown(f"**{label}**{wtxt}: _not applicable_")
        return
    colour = "#ef4444" if value >= 0.60 else "#f59e0b" if value >= 0.30 else "#22c55e"
    st.markdown(
        f"**{label}**{wtxt}: "
        f"<span style='color:{colour};font-size:1.1em;font-weight:700'>{value:.3f}</span>",
        unsafe_allow_html=True,
    )
    st.progress(float(value))


def shap_explanation(model, scaler, row_features, manifest):
    numeric_cols = manifest["numeric_features"]
    try:
        from src.embedding import embed_texts
        text = pd.Series([row_features.get("combined_text", "")])
        emb = embed_texts(text)
        x_numeric = np.array([[row_features.get(c, 0.0) for c in numeric_cols]], dtype=np.float32)
        X = np.hstack([emb, x_numeric])
        X_scaled = scaler.transform(X)

        explainer = shap.LinearExplainer(model, X_scaled, feature_perturbation="interventional")
        shap_vals = explainer.shap_values(X_scaled)

        feature_names = [f"emb_{i}" for i in range(384)] + numeric_cols
        top_idx = np.argsort(np.abs(shap_vals[0]))[-10:][::-1]

        fig, ax = plt.subplots(figsize=(7, 3))
        vals = shap_vals[0][top_idx]
        names = [feature_names[i] for i in top_idx]
        colours = ["#ef4444" if v > 0 else "#22c55e" for v in vals]
        ax.barh(names[::-1], vals[::-1], color=colours[::-1])
        ax.axvline(0, color="#334155", linewidth=0.8)
        ax.set_xlabel("SHAP value (positive = more harmful)")
        ax.set_title("Top 10 feature contributions")
        fig.tight_layout()
        return fig
    except Exception:
        return None


def main():
    st.title("🛡️ Harmful Video Duplicate Detection")
    st.caption("Trust & Safety Analyst Review Dashboard")

    model, scaler, manifest = load_model()

    with st.sidebar:
        st.header("Queue")
        split_name = st.selectbox("Data split", ["test", "val", "train"])
        results_df = load_results(split_name)

        if results_df is None:
            st.warning(f"No inference results found for '{split_name}'. Run run_inference.py first.")
            return

        show_only = st.multiselect(
            "Filter by decision",
            ["ALLOW", "HUMAN REVIEW"],
            default=["HUMAN REVIEW"],
        )
        filtered = results_df[results_df["decision"].isin(show_only)] if show_only else results_df
        st.metric("Videos in queue", len(filtered))

        if len(filtered) == 0:
            st.info("No videos match the current filter.")
            return

        video_id = st.selectbox("Select video", filtered["video_id"].tolist())

    row = results_df[results_df["video_id"] == video_id].iloc[0].to_dict()
    decision = row.get("decision", "UNKNOWN")
    colour = SCORE_COLOURS.get(decision, "#6b7280")

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.subheader(f"Video: `{video_id}`")
    with col2:
        st.markdown(
            f"<div style='background:{colour};color:white;padding:8px 16px;"
            f"border-radius:8px;text-align:center;font-weight:700;font-size:1.1em'>"
            f"{decision}</div>",
            unsafe_allow_html=True,
        )
    with col3:
        mc = row.get("match_confidence")
        if mc is not None and not pd.isna(mc):
            st.metric("Match Confidence", f"{mc:.3f}")

    st.divider()

    col_scores, col_detail = st.columns([1, 1])

    with col_scores:
        st.subheader("Score Breakdown")
        score_bar("Text Similarity", row.get("text_score"), None)
        score_bar("Feature (same-medium)", row.get("feature_score"), None)
        score_bar("Metadata Corroboration", row.get("metadata_score"), None)
        st.divider()
        dt = row.get("duplicate_type")
        st.markdown(f"**Duplicate type:** {dt or '—'}  |  **Lane:** {row.get('lane') or '—'}  "
                    f"|  **Priority:** {row.get('priority') or '—'}")
        if row.get("is_cross_medium"):
            st.markdown(f"**Cross-medium match:** `{row.get('medium_transition')}`")
        st.markdown(f"**Layer 1 harmful probability:** {row.get('harmful_prob', 0):.3f}")
        llm = row.get("llm_score")
        if llm is not None and not pd.isna(llm):
            st.markdown(f"**LLM verification score:** {llm:.3f}")
        st.markdown(f"**Reasoning:** _{row.get('reasoning', '')}_")
        matched = row.get("matched_seed_id")
        if matched:
            st.markdown(f"**Matched seed:** `{matched}` ({row.get('best_pair', '')})")

    with col_detail:
        cm_json = row.get("candidate_matches")
        if cm_json:
            try:
                cands = json.loads(cm_json)
            except (TypeError, ValueError):
                cands = []
            if cands:
                st.subheader(f"Candidate duplicates ({row.get('n_candidates', 0)} above gate)")
                tbl = pd.DataFrame([{
                    "rank": i + 1,
                    "seed": c["seed_id"],
                    "text": round(c["text_score"], 3),
                    "medium pair": c["pair"],
                    "type": c["duplicate_type"] or "— below gate",
                } for i, c in enumerate(cands)])
                st.dataframe(tbl, hide_index=True, use_container_width=True)
                st.caption("Ranked by text similarity. The top row is the primary "
                           "match that drives the decision; the rest are other "
                           "potential duplicates for the reviewer to check.")

        st.subheader("SHAP Explanation")
        row_features = row.copy()
        fig = shap_explanation(model, scaler, row_features, manifest)
        if fig:
            st.pyplot(fig)
        else:
            st.info("SHAP explanation unavailable (run inference first to generate embeddings).")

    # ---- matched text: see WHY these videos matched, not just the IDs/scores ----
    try:
        cands = json.loads(row.get("candidate_matches") or "[]")
    except (TypeError, ValueError):
        cands = []
    if cands:
        texts = load_video_texts()
        qv = texts.get(video_id, {})
        st.divider()
        st.subheader(f"Matched text ({len(cands)} above gate)")
        with st.expander("This video's text (all mediums)", expanded=False):
            st.markdown(f"**title:** {_txt(qv, 'title_text')}")
            st.markdown(f"**ocr:** {_txt(qv, 'ocr_text')}")
            st.markdown(f"**asr:** {_txt(qv, 'asr_text')}")
        st.caption("Left is this video, right is the matched seed; highlighted "
                   "characters are where they differ — that's the evasion. The "
                   "analyst reviews the full cluster and assigns lanes.")

        def render_candidate(i, c):
            qm, cm = _parse_pair(c["pair"])
            sv = texts.get(c["seed_id"], {})
            tag = "cross-medium" if c.get("is_cross_medium") else "same-medium"
            st.markdown(
                f"**#{i} · seed `{c['seed_id']}` · text {c['text_score']:.3f} · "
                f"{c.get('duplicate_type') or 'below gate'} · {tag}**"
            )
            cc1, cc2 = st.columns(2)
            q_html, s_html = diff_pair(_txt(qv, f"{qm}_text"), _txt(sv, f"{cm}_text"))
            with cc1:
                st.caption(f"this video — {qm}")
                st.markdown(q_html, unsafe_allow_html=True)
            with cc2:
                st.caption(f"matched seed — {cm}")
                st.markdown(s_html, unsafe_allow_html=True)

        render_candidate(1, cands[0])  # primary match, always visible
        if len(cands) > 1:
            with st.expander(f"Show the other {len(cands) - 1} match(es)"):
                for i, c in enumerate(cands[1:], start=2):
                    render_candidate(i, c)
                    st.divider()

    st.divider()
    st.subheader("Reviewer Action")
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        if st.button("REMOVE", type="primary", use_container_width=True):
            log_reviewer_verdict(video_id, "REMOVE", "analyst_override")
            st.success("Verdict logged: REMOVE")

    with col_b:
        if st.button("ALLOW", use_container_width=True):
            log_reviewer_verdict(video_id, "ALLOW", "analyst_override")
            st.success("Verdict logged: ALLOW")

    with col_c:
        if st.button("ESCALATE", use_container_width=True):
            log_reviewer_verdict(video_id, "ESCALATE", "analyst_override")
            st.success("Verdict logged: ESCALATE")

    st.caption(f"Reviewed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
