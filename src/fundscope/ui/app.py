"""
app.py — FundScope web UI (Streamlit)

Started by the `fundscope` command (fundscope/ui/launch.py), which opens it in the
browser. Tabs: Ask (chat with the tool-calling agent, with sources, tool calls
and retrieved chunks), Live data (Yahoo Finance), Evaluation (reference runs
judged by an LLM), How it works (architecture).
"""

import json

import altair as alt
import pandas as pd
import streamlit as st

from fundscope import config
from fundscope.live_data import get_etf_live_data, get_price_history
from fundscope.llm import PROVIDERS, SESSION, LLMError, resolve
from fundscope.registry import FUNDS
from fundscope.ui import backend

st.set_page_config(page_title="FundScope — ETF research assistant", page_icon="📑",
                   layout="wide", initial_sidebar_state="expanded")

EXAMPLES = [
    "What is the TER of the iShares Core MSCI World ETF?",
    "Compare the replication methods of the iShares and UBS ETFs",
    "Does the iShares KID report the same ongoing cost as the factsheet?",
    "Quale ETF è più adatto a un investitore di lungo periodo attento ai costi?",
]

# Categorical slots 1–4 of the reference palette (validated, fixed order)
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


# ── Cached resources ───────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_retriever():
    return backend.load_retriever()


@st.cache_data(ttl=3600, show_spinner=False)
def live_data(isin: str, refresh_token: int) -> dict:
    return get_etf_live_data(isin, use_cache=refresh_token == 0)


@st.cache_data(ttl=3600, show_spinner=False)
def price_history(isin: str) -> pd.DataFrame:
    rows = get_price_history(isin)
    return pd.DataFrame(rows, columns=["date", "close"]).assign(date=lambda d: pd.to_datetime(d["date"]))


@st.cache_data(show_spinner=False)
def eval_table() -> pd.DataFrame:
    return pd.DataFrame(backend.eval_rows())


def default_model(provider: str) -> str:
    try:
        return resolve("generator", provider)[1]
    except LLMError:
        return ""


# ── Session state ──────────────────────────────────────────────────────────────

state = st.session_state
state.setdefault("messages", [])          # [{"role", "content", "result"?, "error"?}]
state.setdefault("agent_key", None)
state.setdefault("pending", None)         # question clicked in the examples
state.setdefault("live_refresh", 0)


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.caption("Ask questions about ETF factsheets and KIDs. Answers cite the exact documents they come from.")

    st.subheader("Model")
    env_provider = config.LLM_PROVIDER if config.LLM_PROVIDER in PROVIDERS else PROVIDERS[0]
    provider = st.selectbox("Provider", PROVIDERS, index=PROVIDERS.index(env_provider))
    model = st.text_input("Model", value=default_model(provider),
                          help="Any tool-capable model of the provider. Defaults come from .env.")

    st.subheader("Display")
    show_calls = st.toggle("Show tool calls", value=True)
    show_chunks = st.toggle("Show retrieved chunks", value=False)

    st.subheader("Session")
    st.metric("Estimated cost", f"${SESSION.cost_usd:.4f}",
              help="Sum of the estimated cost of every LLM call since the UI started (LLM_MAX_COST_USD caps it).")
    st.caption(f"{SESSION.calls} LLM call(s) · {SESSION.usage.input_tokens + SESSION.usage.cache_read_tokens:,} "
               f"in / {SESSION.usage.output_tokens:,} out tokens")
    if st.button("New conversation", icon=":material/refresh:", width="stretch"):
        state.messages = []
        if "agent" in state:
            state.agent.reset()
        st.rerun()

    st.subheader("Corpus")
    for fund in FUNDS.values():
        docs = ", ".join(backend.DOC_LABELS.get(d, d) for d in fund.doc_types)
        years = ", ".join(str(y) for y in fund.years)
        st.markdown(f"**{fund.name}**  \n`{fund.isin}` · {docs} ({years})")

    st.divider()
    st.caption(f"⚠️ {backend.DISCLAIMER}")


st.title("📑 FundScope")
st.markdown("Retrieval-augmented research assistant over ETF **factsheets** and **KIDs** — "
            "hybrid search, a tool-calling LLM agent and verifiable citations.")

tab_ask, tab_live, tab_eval, tab_how = st.tabs(
    [":material/forum: Ask", ":material/show_chart: Live data",
     ":material/fact_check: Evaluation", ":material/schema: How it works"])


# ── Ask ────────────────────────────────────────────────────────────────────────

def render_result(message: dict) -> None:
    if message.get("error"):
        st.error(message["error"], icon=":material/error:")
        return
    result = message["result"]
    st.markdown(backend.render_answer(result.answer))

    if result.cited_sources:
        st.markdown("**Sources** " + " ".join(
            backend.source_badge(s.etf_isin, s.doc_type, s.year) for s in result.cited_sources))
    if backend.needs_disclaimer(result.question, result.answer):
        st.caption(f":material/info: {backend.DISCLAIMER}")

    u = result.usage
    cost = f"${result.cost_usd:.4f}" if result.cost_usd is not None else "cost unknown"
    st.caption(f"{result.model} · {result.iterations} LLM call(s) · {len(result.tool_calls)} tool call(s) · "
               f"{u.input_tokens + u.cache_read_tokens:,} in / {u.output_tokens:,} out tokens · {cost}")

    if show_calls and result.tool_calls:
        with st.expander(f"Tool calls ({len(result.tool_calls)})", icon=":material/build:"):
            for call in result.tool_calls:
                status = " — ❌ " + call["error"] if call["error"] else ""
                st.markdown(f"- {backend.tool_call_label(call['name'], call['arguments'])}{status}")
                st.code(json.dumps(call["arguments"], ensure_ascii=False), language="json")
    if show_chunks and result.chunks:
        cited = set(result.cited_chunks)
        with st.expander(f"Retrieved chunks ({len(result.chunks)}, {len(cited)} cited)",
                         icon=":material/description:"):
            for i, c in enumerate(result.chunks, 1):
                mark = "✅ cited" if i in cited else "not cited"
                st.markdown(f"**[Chunk {i}]** {backend.source_badge(c.etf_isin, c.doc_type, c.year)} "
                            f"· {c.section_heading or '—'} · {c.block_type} · similarity {c.score:.3f} · {mark}")
                st.text(c.text)


def get_agent():
    """The agent for the selected provider/model, kept across reruns (it holds the history)."""
    key = (provider, model)
    if state.agent_key != key or "agent" not in state:
        state.agent = None
        state.agent_key = key
        with st.spinner("Loading the search index and the embedding model (about 30 s on the first start)…"):
            retriever = get_retriever()
        try:
            # st.write inside the callback lands in the status box open while the agent runs
            state.agent = backend.build_agent(
                retriever, provider, model,
                on_tool_call=lambda name, args: st.write(backend.tool_call_label(name, args)))
            state.agent_error = None
        except LLMError as e:
            state.agent_error = str(e)
    return state.agent


with tab_ask:
    agent = get_agent()
    if agent is None:
        st.warning(f"**The LLM is not available:** {state.agent_error}  \n"
                   "Copy `.env.example` to `.env`, add the API key and restart `fundscope` "
                   "(check it with `uv run python -m fundscope.llm.cli --ping`). "
                   "The Live data, Evaluation and How it works tabs work without it.",
                   icon=":material/key_off:")

    if not state.messages:
        st.markdown("##### Try one of these")
        cols = st.columns(2)
        for i, example in enumerate(EXAMPLES):
            if cols[i % 2].button(example, width="stretch", disabled=agent is None):
                state.pending = example

    for message in state.messages:
        with st.chat_message(message["role"], avatar="🧑" if message["role"] == "user" else "📑"):
            if message["role"] == "user":
                st.markdown(message["content"].replace("$", r"\$"))
            else:
                render_result(message)

    question = st.chat_input("Ask about costs, risk, replication, holdings, performance…",
                             disabled=agent is None) or state.pending
    if question and agent is not None:
        state.pending = None
        state.messages.append({"role": "user", "content": question})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(question.replace("$", r"\$"))
        with st.chat_message("assistant", avatar="📑"):
            with st.status("Thinking…", expanded=True) as status:
                try:
                    result = agent.run(question)
                    message = {"role": "assistant", "result": result}
                    status.update(label=f"Done — {len(result.tool_calls)} tool call(s)",
                                  state="complete", expanded=False)
                except Exception as e:          # LLMError, BudgetExceededError, network…
                    message = {"role": "assistant", "error": f"{type(e).__name__}: {e}"}
                    status.update(label="Failed", state="error", expanded=False)
            render_result(message)
        state.messages.append(message)
        st.rerun()          # refresh the session cost in the sidebar


# ── Live data ──────────────────────────────────────────────────────────────────

def fmt_pct(value) -> str:
    return f"{value:+.2f}%" if value is not None else "n/a"


def fmt_big(value) -> str:
    if value is None:
        return "n/a"
    for unit, size in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if value >= size:
            return f"{value / size:.2f}{unit}"
    return f"{value:,.0f}"


with tab_live:
    funds = [f for f in FUNDS.values() if f.yahoo_ticker]
    col_pick, col_refresh = st.columns([4, 1], vertical_alignment="bottom")
    fund = col_pick.selectbox("Fund", funds, format_func=lambda f: f"{f.name} ({f.yahoo_ticker})")
    if col_refresh.button("Refresh", icon=":material/refresh:", width="stretch"):
        state.live_refresh += 1

    with st.spinner("Fetching Yahoo Finance data…"):
        data = live_data(fund.isin, state.live_refresh)
    if "error" in data:
        st.warning(f"Live data unavailable for {data.get('ticker', fund.yahoo_ticker)}: {data['error']}",
                   icon=":material/cloud_off:")
    else:
        cur = data.get("currency") or ""
        m = st.columns(4)
        m[0].metric("Price", f"{data['current_price']:,.2f} {cur}")
        m[1].metric("YTD", fmt_pct(data.get("return_ytd")))
        m[2].metric("1 month", fmt_pct(data.get("return_1m_pct")))
        m[3].metric("3 months", fmt_pct(data.get("return_3m_pct")))
        m = st.columns(4)
        low, high = data.get("week52_low"), data.get("week52_high")
        m[0].metric("52-week range", f"{low:,.2f} – {high:,.2f}" if low and high else "n/a")
        m[1].metric("AUM", fmt_big(data.get("total_assets")))
        m[2].metric("Avg daily volume (3m)", fmt_big(data.get("avg_volume")))
        m[3].metric("As of", data["as_of"])

        hist = price_history(fund.isin)
        if not hist.empty:
            chart = (
                alt.Chart(hist)
                .mark_line(strokeWidth=2, color=SERIES_COLORS[0])
                .encode(
                    x=alt.X("date:T", title=None),
                    y=alt.Y("close:Q", title=f"Close ({cur})", scale=alt.Scale(zero=False)),
                    tooltip=[alt.Tooltip("date:T", title="Date"),
                             alt.Tooltip("close:Q", title=f"Close ({cur})", format=",.2f")],
                )
                .properties(height=320, title=f"{fund.yahoo_ticker} — daily close, last 12 months")
            )
            st.altair_chart(chart.interactive(bind_y=False), width="stretch")
        st.caption(f"Source: Yahoo Finance via yfinance ({data['ticker']}), cached for an hour. "
                   "The agent gets the same data through its get_live_data tool.")


# ── Evaluation ─────────────────────────────────────────────────────────────────

with tab_eval:
    table = eval_table()
    if table.empty:
        st.info("No evaluation reports found in evaluation/.")
    else:
        st.markdown(
            "Every run answers the ground-truth questions (17 in English, plus 5 in Italian from phase 5); "
            f"a second model (**{table['judge'].dropna().iloc[0]}**) judges each answer against the reference "
            "for correctness and for groundedness in the retrieved documents.")
        latest = table[table["run"] == "Agent (phase 5)"].iloc[0]
        base = table.iloc[0]
        m = st.columns(4)
        m[0].metric("Judge pass rate — agent", f"{latest['pass']:.0%}",
                    f"{(latest['pass'] - base['pass']) * 100:+.0f} pts vs baseline")
        m[1].metric("Correctness", f"{latest['correctness']:.0%}",
                    f"{(latest['correctness'] - base['correctness']) * 100:+.0f} pts vs baseline")
        m[2].metric("Grounded", f"{latest['grounded']:.0%}",
                    f"{(latest['grounded'] - base['grounded']) * 100:+.0f} pts vs baseline")
        m[3].metric("Cost of a full run + judge", f"${latest['cost_usd']:.2f}")

        metrics = {"pass": "Judge pass", "correctness": "Correctness", "grounded": "Grounded"}
        long = table.melt(id_vars=["run"], value_vars=list(metrics), var_name="metric", value_name="value")
        long["metric"] = long["metric"].map(metrics)
        runs = list(table["run"])
        chart = (
            alt.Chart(long)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X("metric:N", title=None, sort=list(metrics.values()), axis=alt.Axis(labelAngle=0)),
                xOffset=alt.XOffset("run:N", sort=runs, scale=alt.Scale(paddingInner=0.1)),
                y=alt.Y("value:Q", title=None, axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("run:N", sort=runs, title="Run",
                                scale=alt.Scale(domain=runs, range=SERIES_COLORS[:len(runs)]),
                                legend=alt.Legend(orient="top")),
                tooltip=[alt.Tooltip("run:N", title="Run"), alt.Tooltip("metric:N", title="Metric"),
                         alt.Tooltip("value:Q", title="Value", format=".1%")],
            )
            .properties(height=340)
        )
        st.altair_chart(chart, width="stretch")

        st.dataframe(
            table.drop(columns=["judge"]),
            hide_index=True, width="stretch",
            column_config={
                "run": "Run", "engine": "Engine", "questions": "Questions", "model": "Generator",
                "pass": st.column_config.ProgressColumn("Judge pass", format="percent", min_value=0, max_value=1),
                "correctness": st.column_config.ProgressColumn("Correctness", format="percent",
                                                               min_value=0, max_value=1),
                "grounded": st.column_config.ProgressColumn("Grounded", format="percent", min_value=0, max_value=1),
                "attribution": st.column_config.NumberColumn("Attribution", format="%.2f"),
                "retrieval_precision": st.column_config.NumberColumn("Retrieval P", format="%.2f"),
                "retrieval_recall": st.column_config.NumberColumn("Retrieval R", format="%.2f"),
                "cost_usd": st.column_config.NumberColumn("Cost (run + judge)", format="$%.2f"),
            },
        )
        st.caption(
            "The baseline mostly failed by abstaining: the right document was retrieved, but not the chunk with "
            "the value. The agent searches again with other wording, which fixed most of those. On this small "
            "corpus (~15k tokens) putting every document in the prompt is better still; retrieval is meant to pay "
            "off as the corpus grows. Retrieval precision drops for the agent and full context because they read "
            "more documents than the minimum a question needs.")


# ── How it works ───────────────────────────────────────────────────────────────

with tab_how:
    st.graphviz_chart("""
    digraph {
        rankdir=LR; bgcolor="transparent";
        node [shape=box, style="rounded,filled", fillcolor="#eef3fb", color="#2a78d6",
              fontname="Helvetica", fontsize=11, margin="0.2,0.1"];
        edge [color="#8a8984", fontname="Helvetica", fontsize=9, fontcolor="#8a8984"];
        pdf    [label="PDF factsheets\\n+ KIDs"];
        ingest [label="ingest\\npdfplumber, tables,\\nkey-facts chunk"];
        index  [label="ChromaDB + BM25\\nmultilingual-e5-base"];
        agent  [label="Tool-calling agent\\nClaude Haiku 4.5", fillcolor="#fdf0ea", color="#eb6834"];
        live   [label="Yahoo Finance\\nlive data"];
        answer [label="Answer with\\nresolved citations"];
        judge  [label="Evaluation\\nLLM judge + heuristics", fillcolor="#e9f7f1", color="#1baf7a"];
        pdf -> ingest -> index;
        index -> agent [label="search_etf_docs", dir=both];
        live -> agent [label="get_live_data"];
        agent -> answer -> judge;
    }
    """, width="stretch")

    left, right = st.columns(2)
    left.markdown("""
**Retrieval**
- Chunks of ~1000 characters (tables kept whole) plus a synthetic *key facts* chunk per document.
- Hybrid search: multilingual embeddings + BM25, fused with Reciprocal Rank Fusion — BM25 catches
  ISINs, acronyms (TER, SRI) and figures that embeddings rank poorly.
- Metadata filters per fund, document type and year; *comparative* and *cross-document* modes so that
  one fund cannot crowd out the other.

**Agent**
- The LLM picks its tools and arguments (funds, document type, mode) and searches again when the
  first results miss the answer.
- It cites `[Chunk N]`; the code rewrites each tag with the chunk's real metadata, so a citation can
  only point to a document that was actually retrieved.
""")
    right.markdown("""
**Provider-neutral LLM layer**
- One `LLMClient` interface: native Anthropic backend (prompt caching) and an OpenAI-compatible one
  (Groq, Gemini, OpenRouter) — switching provider is a `.env` change.
- Every call is priced; a per-run budget stops runaway costs.

**Evaluation and engineering**
- 22 ground-truth questions (English and Italian), scored by heuristics and an LLM judge.
- 60+ pytest tests with no network or API key (LLM and embedding model mocked), ruff, GitHub Actions CI.
- Installable package (`uv`), one command to start this UI.
""")
