"""
PulseAI dashboard. A deliberately THIN client: every bit of actual logic
lives in core/pipeline.py's run_full_pipeline. This file only renders a
DashboardPayload and reacts to a couple of inputs.

Run with: streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

SRC_DIR = Path(__file__).parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pulseai.ai.embeddings import EmbeddingClient  # noqa: E402
from pulseai.ai.llm_client import LLMClient  # noqa: E402
from pulseai.core.pipeline import run_full_pipeline  # noqa: E402
from pulseai.schemas.report import DashboardPayload  # noqa: E402
from pulseai.storage.db import get_connection  # noqa: E402

URGENCY_COLORS = {
    "critical": "#DC2626",
    "high": "#F97316",
    "medium": "#EAB308",
    "low": "#16A34A",
}
SENTIMENT_COLORS = {
    "negative": "#DC2626",
    "mixed": "#A855F7",
    "neutral": "#94A3B8",
    "positive": "#16A34A",
}
URGENCY_ORDER = ["low", "medium", "high", "critical"]

DB_PATH = "pulseai.db"
DEFAULT_SAMPLE_CSV = "sample_data/feedback_sample.csv"


@st.cache_resource
def get_llm_client() -> LLMClient | None:
    try:
        return LLMClient()
    except RuntimeError:
        return None


@st.cache_resource
def get_embedding_client() -> EmbeddingClient | None:
    try:
        return EmbeddingClient()
    except RuntimeError:
        return None


def get_db_connection():
    return get_connection(DB_PATH)


def render_header(payload: DashboardPayload) -> None:
    st.title("PulseAI")
    st.caption(
        f"Customer feedback intelligence · week of "
        f"{payload.aggregate.period_start} to {payload.aggregate.period_end} · "
        f"generated {payload.generated_at.strftime('%b %d, %Y %H:%M')}"
    )

    cols = st.columns(4)
    cols[0].metric("Total feedback", payload.aggregate.total_feedback)
    cols[1].metric("Needs human review", payload.aggregate.review_queue_count)
    critical_high = payload.aggregate.urgency_distribution.get("critical", 0) + \
        payload.aggregate.urgency_distribution.get("high", 0)
    cols[2].metric("Critical + high urgency", critical_high)
    cols[3].metric("Themes identified", len(payload.themes))


def render_executive_summary(payload: DashboardPayload, quote_lookup: dict[str, str]) -> None:
    st.subheader("📋 Weekly Executive Brief")
    st.markdown(f"**{payload.summary.headline}**")

    if payload.summary.key_insights:
        st.markdown("##### Key insights")
        for insight in payload.summary.key_insights:
            with st.expander(
                f"{insight.statement}  ·  confidence {insight.confidence:.0%}"
            ):
                st.caption("Grounded in these actual feedback items:")
                for fid in insight.supporting_feedback_ids:
                    quote = quote_lookup.get(fid, "(quote not found)")
                    st.markdown(f"> \"{quote}\"  \n`{fid}`")

    if payload.summary.recommended_actions:
        st.markdown("##### Recommended actions")
        for action in sorted(payload.summary.recommended_actions, key=lambda a: a.priority):
            st.markdown(f"**{action.priority}. {action.action}**")
            st.caption(f"{action.rationale} — *Expected impact:* {action.expected_impact}")

    if payload.summary.watch_items:
        st.markdown("##### Watch")
        for item in payload.summary.watch_items:
            st.markdown(f"- {item}")

    st.info(f"**Caveats:** {payload.summary.caveats}")


def render_distribution_charts(payload: DashboardPayload) -> None:
    st.subheader("What does this week's feedback look like?")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Where are the problems concentrated?**")
        cat_counts = payload.aggregate.category_distribution
        if cat_counts:
            df = pd.DataFrame(
                {"category": [c.value.replace("_", " ").title() for c in cat_counts],
                 "count": list(cat_counts.values())}
            ).sort_values("count", ascending=True)
            fig = px.bar(df, x="count", y="category", orientation="h",
                         color_discrete_sequence=["#2563EB"])
            fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0),
                               xaxis_title="Feedback items", yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**How does the customer base feel overall?**")
        sent_counts = payload.aggregate.sentiment_distribution
        if sent_counts:
            df = pd.DataFrame(
                {"sentiment": [s.value for s in sent_counts],
                 "count": list(sent_counts.values())}
            )
            fig = px.bar(
                df, x="sentiment", y="count", color="sentiment",
                color_discrete_map=SENTIMENT_COLORS,
            )
            fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0),
                               showlegend=False, xaxis_title="", yaxis_title="Feedback items")
            st.plotly_chart(fig, use_container_width=True)

    with col3:
        st.markdown("**How much is on fire right now?**")
        urg_counts = payload.aggregate.urgency_distribution
        if urg_counts:
            ordered = [(u, urg_counts.get(u, 0)) for u in URGENCY_ORDER if u in urg_counts]
            df = pd.DataFrame({"urgency": [u for u, _ in ordered], "count": [c for _, c in ordered]})
            fig = px.bar(
                df, x="urgency", y="count", color="urgency",
                color_discrete_map=URGENCY_COLORS,
                category_orders={"urgency": URGENCY_ORDER},
            )
            fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0),
                               showlegend=False, xaxis_title="", yaxis_title="Feedback items")
            st.plotly_chart(fig, use_container_width=True)


def render_themes(payload: DashboardPayload) -> None:
    st.subheader("What are today's top specific issues?")
    if not payload.aggregate.top_themes:
        st.caption("No themes identified for this period.")
        return

    df = pd.DataFrame([
        {"label": t.label, "priority_score": t.priority_score, "size": t.size}
        for t in payload.aggregate.top_themes
    ]).sort_values("priority_score", ascending=True)

    fig = px.bar(
        df, x="priority_score", y="label", orientation="h",
        color_discrete_sequence=["#0F172A"],
        text="size",
    )
    fig.update_traces(texttemplate="%{text} items", textposition="outside")
    fig.update_layout(
        height=80 + 60 * len(df), margin=dict(l=0, r=60, t=10, b=0),
        xaxis_title="Priority score (volume × urgency × negativity, normalized)",
        yaxis_title="",
    )
    st.plotly_chart(fig, use_container_width=True)

    for theme in payload.aggregate.top_themes:
        with st.expander(f"{theme.label} — {theme.size} item(s), priority {theme.priority_score:.2f}"):
            st.write(theme.description)
            for q in theme.representative_quotes:
                st.markdown(f"> \"{q.text}\"  \n`{q.feedback_id}`")


def render_drilldown(payload: DashboardPayload) -> None:
    st.subheader("Explore every item")

    rows = [{
        "feedback_id": item.feedback_id,
        "category": item.primary_category.value,
        "sentiment": item.sentiment.value,
        "urgency": item.urgency.value,
        "category_confidence": round(item.category_confidence, 2),
        "sentiment_confidence": round(item.sentiment_confidence, 2),
        "needs_human_review": item.needs_human_review,
        "quote": item.key_quote,
    } for item in payload.items]
    df = pd.DataFrame(rows)

    tab_all, tab_review = st.tabs(["All feedback", f"⚠️ Review queue ({payload.aggregate.review_queue_count})"])

    with tab_all:
        categories = st.multiselect("Filter by category", sorted(df["category"].unique()), key="cat_filter")
        filtered = df[df["category"].isin(categories)] if categories else df
        st.dataframe(filtered, use_container_width=True, hide_index=True)

    with tab_review:
        review_df = df[df["needs_human_review"]].sort_values("category_confidence")
        if review_df.empty:
            st.caption("Nothing flagged for review this period.")
        else:
            st.dataframe(review_df.drop(columns=["needs_human_review"]),
                         use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(page_title="PulseAI", page_icon="📊", layout="wide")

    llm_client = get_llm_client()
    embed_client = get_embedding_client()

    with st.sidebar:
        st.header("PulseAI")
        st.caption("Customer feedback intelligence")

        if llm_client is None:
            st.error(
                "No OPENAI_API_KEY found. Copy `.env.example` to `.env` and "
                "add your key, then restart the app."
            )

        csv_path = st.text_input("Feedback CSV path", value=DEFAULT_SAMPLE_CSV)
        run_clicked = st.button("Run weekly analysis", type="primary", disabled=llm_client is None)
        st.caption(
            f"Model: `{llm_client.config.model}`" if llm_client else "Model: (not configured)"
        )

    if run_clicked and llm_client is not None and embed_client is not None:
        conn = get_db_connection()
        with st.status("Running analysis pipeline...", expanded=True) as status:
            st.write("Ingesting and validating feedback...")
            status.update(label="Analyzing feedback (this calls the LLM per item)...")
            try:
                payload = run_full_pipeline(csv_path, llm_client, embed_client, conn)
            except FileNotFoundError:
                status.update(label="Failed", state="error")
                st.error(f"Could not find file: {csv_path}")
                return
            except Exception as exc:  # noqa: BLE001
                status.update(label="Failed", state="error")
                st.error(f"Pipeline run failed: {exc}")
                return
            status.update(label="Done", state="complete")
        st.session_state["payload"] = payload

    payload: DashboardPayload | None = st.session_state.get("payload")

    if payload is None:
        st.info("Set a CSV path in the sidebar and click **Run weekly analysis** to get started.")
        return

    quote_lookup = {item.feedback_id: item.key_quote for item in payload.items}

    render_header(payload)
    st.divider()
    render_executive_summary(payload, quote_lookup)
    st.divider()
    render_distribution_charts(payload)
    st.divider()
    render_themes(payload)
    st.divider()
    render_drilldown(payload)


main()