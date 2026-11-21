"""Budget simulator: how many *extra* conversions does each targeting rule buy?

Reads only the small precomputed artifacts (curves.csv, deciles.csv, metrics.json), so it runs
anywhere, including a free Hugging Face Space, without the 14M-row dataset or the model.
"""

from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

HERE = Path(__file__).resolve().parent
ART = next(p for p in (HERE / "artifacts", HERE.parent / "artifacts") if p.exists())

LABELS = {
    "random": "Random",
    "response": "Response model",
    "s_learner": "S-learner",
    "t_learner": "T-learner",
    "x_learner": "X-learner",
    "transformed_outcome": "Transformed outcome",
}

st.set_page_config(page_title="Uplift budget simulator", layout="wide")


@st.cache_data
def load():
    curves = pd.read_csv(ART / "curves.csv")
    deciles = pd.read_csv(ART / "deciles.csv")
    summary = json.loads((ART / "metrics.json").read_text())
    return curves, deciles, summary


curves, deciles, summary = load()
outcome = "conversion"
prod = summary.get("production_model")
c = curves[(curves.outcome == outcome) & (curves.kind == "uplift")].copy()
c["model"] = c["model"].map(LABELS).fillna(c["model"])

st.title("Who should get the voucher?")
st.caption(
    "Criteo Uplift v2.1 test split, "
    f"{summary['meta']['split']['test']:,} customers. Curves are measured, not modelled: "
    "incremental conversions = (treated rate - control rate) among the customers targeted. "
    "Data: Criteo AI Lab, CC BY-NC-SA 4.0."
)

with st.sidebar:
    st.header("Campaign")
    base = st.number_input("Customer base", 10_000, 50_000_000, 1_000_000, step=100_000)
    budget = st.slider("Share of base that gets the voucher", 1, 100, 20, format="%d%%") / 100
    cost = st.number_input("Cost per voucher sent", 0.0, 1_000.0, 0.05, step=0.01)
    value = st.number_input("Margin per incremental conversion", 0.0, 100_000.0, 60.0, step=5.0)

at = c[c.share.round(2) == round(budget, 2)].copy()
at["incremental"] = at["value"] * base
at["profit"] = at["incremental"] * value - cost * base * budget
at = at.sort_values("incremental", ascending=False)

best = at.iloc[0]
rand = at[at.model == "Random"].iloc[0]
col1, col2, col3 = st.columns(3)
col1.metric(f"Best rule: {best.model}", f"{best.incremental:,.0f} extra conversions")
col2.metric("Random targeting", f"{rand.incremental:,.0f}")
col3.metric(
    "Profit difference vs random",
    f"{best.profit - rand.profit:,.0f}",
    help="(incremental conversions x margin) - (vouchers x cost)",
)

st.subheader("Incremental conversions by share targeted")
chart_df = c.assign(incremental=c["value"] * base)
line = (
    alt.Chart(chart_df)
    .mark_line()
    .encode(
        x=alt.X("share:Q", axis=alt.Axis(format="%"), title="Share of base targeted"),
        y=alt.Y("incremental:Q", title="Incremental conversions"),
        color=alt.Color("model:N", title=None),
        tooltip=[
            "model",
            alt.Tooltip("share:Q", format=".0%"),
            alt.Tooltip("incremental:Q", format=",.0f"),
        ],
    )
)
rule = alt.Chart(pd.DataFrame({"share": [budget]})).mark_rule(strokeDash=[4, 4]).encode(x="share:Q")
st.altair_chart(line + rule, use_container_width=True)

st.subheader(f"At {budget:.0%} of the base")
st.dataframe(
    at[["model", "incremental", "profit"]].rename(
        columns={"model": "Rule", "incremental": "Incremental conversions", "profit": "Profit"}
    ),
    hide_index=True,
    use_container_width=True,
    column_config={
        "Incremental conversions": st.column_config.NumberColumn(format="%.0f"),
        "Profit": st.column_config.NumberColumn(format="%.0f"),
    },
)

if prod:
    st.subheader(f"Observed uplift by decile ({LABELS.get(prod, prod)}, test split)")
    d = deciles[deciles.outcome == outcome].copy()
    d["uplift_pp"] = d["uplift"] * 100
    bars = (
        alt.Chart(d)
        .mark_bar()
        .encode(
            x=alt.X("decile:O", title="Decile (1 = highest predicted uplift)"),
            y=alt.Y("uplift_pp:Q", title="Treated - control conversion rate (pp)"),
            tooltip=["decile", "n", alt.Tooltip("uplift_pp:Q", format=".3f")],
        )
    )
    st.altair_chart(bars, use_container_width=True)
    st.caption(
        "Deciles at the bottom with uplift near or below zero are customers the voucher does not "
        "move (or puts off): skipping them saves budget."
    )
