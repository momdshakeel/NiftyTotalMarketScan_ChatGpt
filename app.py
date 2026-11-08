"""
NSE 500 RS Rating + Minervini Trend Template + VCP / breakout / Stage scanner - Streamlit app.

Run locally:   pip install -r requirements.txt && streamlit run app.py
Deploy free:   push this folder to GitHub -> share.streamlit.io -> New app -> app.py
Files needed in the same folder: app.py, ui.py, nse500_rs_rating.py, requirements.txt
"""
import io
import os
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nse500_rs_rating as m  # noqa: E402
import ui  # noqa: E402

CACHE_HOURS = 6
IST = timezone(timedelta(hours=5, minutes=30))

st.set_page_config(page_title="NSE 500 RS Scanner", page_icon="📈", layout="wide")

# ---------------- Appearance (first, so the CSS can use the chosen font) ----------------
with st.sidebar:
    with st.expander("🎨 Appearance", expanded=False):
        font = st.selectbox("Font", list(ui.FONTS), index=0)
        density = st.radio("Table density", ["Comfortable", "Compact"], horizontal=True)
        st.caption("Light / dark: open the ⋮ menu (top right) → Settings → Theme. "
                   "The app follows your device by default.")
ui.inject_css(font)


# ---------------- Data loading (cached, shared by all visitors) ----------------
def parse_universe(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip() for c in df.columns]
    df = df[["Company Name", "Industry", "Symbol"]].copy()
    df["Ticker"] = df["Symbol"].str.strip() + ".NS"
    return df


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def fetch_universe():
    """Nifty 500 list: NSE download -> local ind_nifty500list.csv -> None."""
    try:
        r = requests.get(m.NSE500_URL, timeout=30,
                         headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        r.raise_for_status()
        return parse_universe(pd.read_csv(io.StringIO(r.text)))
    except Exception:
        local = os.path.join(HERE, "ind_nifty500list.csv")
        if os.path.exists(local):
            return parse_universe(pd.read_csv(local))
        return None


@st.cache_data(ttl=CACHE_HOURS * 3600, show_spinner=False)
def compute(meta: pd.DataFrame):
    md = m.download_market_data(meta["Ticker"].tolist())
    prices = md["Close"]
    bench = m.get_benchmark()
    result, as_of = m.build_table(prices, bench, meta, ohlcv=md)
    return result, as_of, datetime.now(IST), prices.shape[1]


meta = fetch_universe()
if meta is None:
    st.warning("Could not download the Nifty 500 list from NSE. Upload `ind_nifty500list.csv` "
               "(from niftyindices.com), or add it to the app folder.")
    up = st.file_uploader("ind_nifty500list.csv", type="csv")
    if up is None:
        st.stop()
    meta = parse_universe(pd.read_csv(up))

try:
    with st.spinner("Downloading ~500 stocks from Yahoo Finance (first load takes 1-2 minutes)..."):
        result, as_of, fetched_at, n_prices = compute(meta)
except Exception as e:
    st.error(f"Data download failed: {e}")
    if st.button("Retry"):
        st.rerun()
    st.stop()

# ---------------- Sidebar filters ----------------
with st.sidebar:
    st.subheader("Filters")
    min_rs = st.slider("Minimum RS Rating", 1, 99, 1)
    pref_rs = st.slider("RS for 'Top picks'", 50, 99, m.PREFERRED_RS)
    rsi_min = st.slider("RSI above (Top picks)", 0, 80, 50)
    industries = sorted(result["Industry"].dropna().unique())
    pick_ind = st.multiselect("Industry", industries)
    q = st.text_input("Search symbol / company")
    st.divider()
    st.caption(f"Prices as of **{as_of.date()}**  \nFetched {fetched_at:%d %b %Y, %H:%M} IST  \n"
               f"Auto-refreshes every {CACHE_HOURS} hours")
    if st.button("🔄 Refresh data now", use_container_width=True):
        compute.clear()
        st.rerun()

# ---------------- Apply filters ----------------
df = result[result["RS_Rating"] >= min_rs]
if pick_ind:
    df = df[df["Industry"].isin(pick_ind)]
if q.strip():
    s = q.strip().lower()
    df = df[df["Symbol"].str.lower().str.contains(s, regex=False)
            | df["Company Name"].str.lower().str.contains(s, regex=False, na=False)]

top = df[df.TT_Pass & (df.RS_Rating >= pref_rs) & df.RS_Line_Near_High & (df.RSI14 > rsi_min)]
tt = df[df.TT_Pass]
blue = df[df.RS_Blue_Dot & (df.RS_Rating >= pref_rs)]
vcp = df[df.VCP_Setup].sort_values("Pct_To_Pivot")
brk = df[df.Vol_Breakout].sort_values(["Breakout_Days_Ago", "Breakout_Vol_x"], ascending=[True, False])
s1 = df[df.Stage1_Base].sort_values("Base_Days", ascending=False)

# ---------------- Header + KPI cards ----------------
ui.hero("RS Rating Scanner",
        "Relative-strength ranking, Minervini Trend Template, VCP setups, volume breakouts and "
        "Weinstein stages for every Nifty 500 stock.",
        [f"<b>As of</b> {as_of:%d %b %Y}", f"<b>Fetched</b> {fetched_at:%d %b, %H:%M} IST",
         f"<b>{len(result)}</b> stocks rated"])
ui.kpis([
    ("Top picks", len(top), f"RS ≥ {pref_rs} + trend + RS line", "#10b981"),
    ("Trend Template", len(tt), "all 8 conditions", "#6366f1"),
    ("VCP setups", len(vcp), "tight & near pivot", "#8b5cf6"),
    ("Volume breakouts", len(brk), f"last {m.BREAKOUT_RECENT_DAYS} sessions", "#f97316"),
    ("Stage 1 bases", len(s1), "early watch-list", "#3b82f6"),
    ("RS blue dot", len(blue), "RS line leads price", "#ec4899"),
])


def show(frame, name, text, keys=None, sort=("RS_Rating", -1)):
    ui.note(text)
    ui.table(frame, keys or ui.COLS_DEFAULT, sort, font, density)
    if not frame.empty:
        st.download_button("⬇️ Download CSV", frame.to_csv(index=False).encode("utf-8"),
                           file_name=f"nse500_{name}_{as_of.date()}.csv", mime="text/csv",
                           key=f"dl_{name}")


def show_industries():
    ui.note("Industries ranked by a blend of <b>median RS Rating (50%)</b>, <b>share of stocks with RS ≥ 80 "
            "(25%)</b> and <b>share passing the Trend Template (25%)</b>. Uses the whole NSE 500 - "
            "sidebar filters don't apply.")
    min_n = st.slider("Minimum stocks per industry", 1, 10, 3,
                      help="Small industries give noisy scores, so they are hidden below this size.")
    ind = m.industry_strength(result, min_stocks=min_n)
    if ind.empty:
        st.info("No industries meet the minimum size.")
        return
    ui.table(ind, ui.COLS_INDUSTRY, ("Rank", 1), font, density)
    st.download_button("⬇️ Download CSV", ind.to_csv(index=False).encode("utf-8"),
                       file_name=f"nse500_industry_strength_{as_of.date()}.csv", mime="text/csv",
                       key="dl_industry_table")
    st.markdown("#### Stocks in an industry")
    pick = st.selectbox("Choose an industry (listed by strength)", ind["Industry"].tolist())
    members = result[result["Industry"] == pick].sort_values("RS_Rating", ascending=False)
    show(members, "industry_stocks", f"All rated stocks in <b>{pick}</b>, sorted by RS Rating.")


(tab_top, tab_tt, tab_vcp, tab_brk, tab_s1, tab_blue, tab_ind, tab_all) = st.tabs([
    f"Top picks · {len(top)}", f"Trend Template · {len(tt)}", f"VCP setups · {len(vcp)}",
    f"Breakouts · {len(brk)}", f"Stage 1 bases · {len(s1)}", f"RS blue dot · {len(blue)}",
    "Industries", f"All stocks · {len(df)}"])

with tab_top:
    show(top, "top_picks", f"Passes all 8 Trend Template conditions, <b>RS ≥ {pref_rs}</b>, RS line within 5% "
                           f"of its 52-week high and <b>RSI &gt; {rsi_min}</b>.")
with tab_tt:
    show(tt, "trend_template", "All 8 Minervini conditions: price above the 50/150/200 SMA, 150 &gt; 200 SMA, "
                               "200 SMA rising, 30% above the 52-week low, within 25% of the 52-week high, "
                               "RS ≥ 70.")
with tab_vcp:
    show(vcp, "vcp_setups",
         "<b>Stage 2</b> stocks with 2+ successively tighter pullbacks (last one ≤ 10%), volume dried up over the "
         f"last 10 days (≤ {int(m.VOL_DRYUP_RATIO * 100)}% of the 50-day average) and price "
         f"0-{m.VCP_NEAR_PIVOT:.0f}% below the <b>pivot</b>. Watch for a close above the pivot on heavy "
         "volume. Closest to pivot first.", ui.COLS_PATTERN, ("Pct_To_Pivot", 1))
with tab_brk:
    show(brk, "volume_breakouts",
         f"Closed above the prior {m.BREAKOUT_LOOKBACK}-session high on <b>≥ {m.BREAKOUT_VOL_X}× the 50-day average "
         f"volume</b> in the last {m.BREAKOUT_RECENT_DAYS} sessions, after a base no deeper than "
         f"{int(m.MAX_BASE_DEPTH * 100)}%, closing in the upper half of the day's range.",
         ui.COLS_PATTERN, ("Breakout_Days_Ago", 1))
with tab_s1:
    show(s1, "stage1_bases",
         "<b>Stage 1</b> = price around a flat 150-day (30-week) average after a decline, with a sideways base of "
         f"{m.BASE_WINDOWS[0]}+ sessions forming (depth ≤ {int(m.MAX_BASE_DEPTH * 100)}%). Early candidates: "
         "they only become buyable after a breakout into Stage 2. Longest bases first.",
         ui.COLS_PATTERN, ("Base_Days", -1))
with tab_blue:
    show(blue, "blue_dot", f"RS line at a 52-week high while price is still more than 2% below its own high "
                           f"(RS ≥ {pref_rs}).")
with tab_ind:
    show_industries()
with tab_all:
    show(df.sort_values("RS_Rating", ascending=False), "all",
         "Every rated stock. <b>Click any column header to sort.</b>")

with st.expander("Notes & limitations"):
    st.markdown(
        "- Stocks with less than 12 months of price history are not rated.\n"
        "- 52-week high/low use closing prices, not intraday extremes.\n"
        "- RS Rating is relative to the NSE 500 only, so it can differ from screeners that rank a wider universe.\n"
        "- **Stages** use the 150-day SMA (about the 30-week average): Stage 2 = price above a rising average, "
        "Stage 4 = below a falling one; Stage 1/3 = flat average after a decline/advance. Stocks with under ~13 "
        "months of history show 'n/a'.\n"
        "- **VCP, bases and breakouts are rule-based approximations** computed on daily closes/highs/lows. "
        "They will flag some patterns a human would reject and miss others, so always confirm on the chart.\n"
        "- Yahoo Finance data can have gaps; a few tickers may be missing on any given day.\n"
        "- Educational tool, not investment advice."
    )
