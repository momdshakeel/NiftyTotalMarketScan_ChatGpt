"""
Nifty Total Market (Nifty 500 + Microcap 250) RS Rating scanner - Streamlit app.
RS Rating, Minervini Trend Template, VCP setups, volume breakouts, Weinstein stages,
liquidity flag, market breadth and sector / industry strength.

Run locally:   pip install -r requirements.txt && streamlit run app.py
Deploy free:   push this folder to GitHub -> share.streamlit.io -> New app -> app.py
Files needed in the same folder: app.py, ui.py, nse500_rs_rating.py, requirements.txt
Optional:      nse_classification.csv (NSE industry / basic industry), ind_niftytotalmarket_list.csv
"""
import os
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nse500_rs_rating as m  # noqa: E402
import ui  # noqa: E402

CACHE_HOURS = 6
IST = timezone(timedelta(hours=5, minutes=30))
LIQ_TEXT = f"avg {m.LIQUIDITY_DAYS}-day traded value ≥ ₹{m.MIN_TRADED_VALUE_CR:.0f} crore"

st.set_page_config(page_title="Total Market RS Scanner", page_icon="📈", layout="wide")

# ---------------- Appearance (first, so the CSS can use the chosen font) ----------------
with st.sidebar:
    with st.expander("🎨 Appearance", expanded=False):
        font = st.selectbox("Font", list(ui.FONTS), index=0)
        density = st.radio("Table density", ["Comfortable", "Compact"], horizontal=True)
        st.caption("Light / dark: open the ⋮ menu (top right) → Settings → Theme. "
                   "The app follows your device by default.")
ui.inject_css(font)


# ---------------- Data loading (cached, shared by all visitors) ----------------
@st.cache_data(ttl=24 * 3600, show_spinner=False)
def fetch_universe():
    """Nifty Total Market list: NSE -> local ind_niftytotalmarket_list.csv -> Nifty 500 + Microcap 250."""
    return m.load_universe(local_file=os.path.join(HERE, m.LOCAL_UNIVERSE_FILE))


@st.cache_data(ttl=7 * 24 * 3600, show_spinner=False)
def load_shares(tickers: tuple):
    """Shares outstanding from Yahoo (they change rarely, so cached for a week)."""
    return m.download_shares(list(tickers))


@st.cache_data(ttl=CACHE_HOURS * 3600, show_spinner=False)
def compute(meta: pd.DataFrame, shares: pd.Series):
    md = m.download_market_data(meta["Ticker"].tolist())
    prices = md["Close"]
    bench = m.get_benchmark()
    result, as_of = m.build_table(prices, bench, meta, ohlcv=md, shares=shares)
    return result, as_of, datetime.now(IST), prices.shape[1]


meta = fetch_universe()
if meta is None:
    st.warning("Could not download the Nifty Total Market list from NSE. Upload "
               "`ind_niftytotalmarket_list.csv` (nseindia.com → Indices → Nifty Total Market → "
               "Download list), or add it to the app folder.")
    up = st.file_uploader("ind_niftytotalmarket_list.csv", type="csv")
    if up is None:
        st.stop()
    meta = m.parse_universe(pd.read_csv(up))
meta = m.attach_classification(meta, m.load_classification(os.path.join(HERE, m.CLASSIFICATION_FILE)))

try:
    with st.spinner("Downloading ~750 stocks from Yahoo Finance (first load takes 3-5 minutes)..."):
        if st.session_state.get("shares_failed"):             # Yahoo throttled us: retry on Refresh only
            shares = pd.Series(float("nan"), index=meta["Ticker"])
        else:
            shares = load_shares(tuple(meta["Ticker"]))
            if shares.notna().mean() < 0.5:                   # don't keep a bad result for a week
                load_shares.clear()
                st.session_state["shares_failed"] = True
        result, as_of, fetched_at, n_prices = compute(meta, shares)
except Exception as e:
    st.error(f"Data download failed: {e}")
    if st.button("Retry"):
        st.rerun()
    st.stop()

has_cls = bool(result["Basic Industry"].notna().any()) if "Basic Industry" in result.columns else False
n_liquid = int(result["Liquid"].sum())

# ---------------- Sidebar filters ----------------
with st.sidebar:
    st.subheader("Filters")
    liquid_only = st.toggle("Liquid stocks only", value=False,
                            help=f"Liquid = {LIQ_TEXT}. Applies to the stock lists. Breadth, sector and "
                                 f"industry figures always use liquid stocks.")
    cap_ok = bool(result["Market_Cap_Cr"].notna().mean() >= 0.5)
    min_cap = st.number_input("Market cap above (₹ crore)", min_value=0, value=500, step=100,
                              disabled=not cap_ok,
                              help="Market cap = shares outstanding × latest close (Yahoo Finance). "
                                   "Set 0 to switch the filter off. Applies to the stock lists; breadth, "
                                   "sector and industry figures don't use it.")
    incl_unknown = st.checkbox("Include unknown market cap", value=False, disabled=not cap_ok,
                               help="Stocks for which Yahoo has no share count.")
    if cap_ok:
        n_cap = int((result["Market_Cap_Cr"] > min_cap).sum())
        n_unk = int(result["Market_Cap_Cr"].isna().sum())
        st.caption(f"{n_cap} stocks above ₹{min_cap:,} cr · {n_unk} with unknown market cap")
    else:
        st.caption("⚠️ Market cap data unavailable from Yahoo right now, so this filter is off. "
                   "Click *Refresh data now* to retry.")
    min_rs = st.slider("Minimum RS Rating", 1, 99, 1)
    pref_rs = st.slider("RS for 'Top picks'", 50, 99, m.PREFERRED_RS)
    rsi_min = st.slider("RSI above (Top picks)", 0, 80, 50)
    pick_sector = st.multiselect("Sector", sorted(result["Sector"].dropna().unique()))
    pick_basic = (st.multiselect("Basic industry", sorted(result["Basic Industry"].dropna().unique()))
                  if has_cls else [])
    q = st.text_input("Search symbol / company")
    st.divider()
    st.caption(f"Prices as of **{as_of.date()}**  \nFetched {fetched_at:%d %b %Y, %H:%M} IST  \n"
               f"Auto-refreshes every {CACHE_HOURS} hours")
    if st.button("🔄 Refresh data now", use_container_width=True):
        compute.clear()
        fetch_universe.clear()
        st.session_state.pop("shares_failed", None)
        st.rerun()

# ---------------- Apply filters ----------------
df = result
if liquid_only:
    df = df[df["Liquid"]]
if cap_ok and min_cap > 0:
    keep = df["Market_Cap_Cr"] > min_cap
    if incl_unknown:
        keep |= df["Market_Cap_Cr"].isna()
    df = df[keep]
df = df[df["RS_Rating"] >= min_rs]
if pick_sector:
    df = df[df["Sector"].isin(pick_sector)]
if pick_basic:
    df = df[df["Basic Industry"].isin(pick_basic)]
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
        "Relative-strength ranking, Minervini Trend Template, VCP setups, volume breakouts, Weinstein "
        "stages, market breadth and sector strength across the Nifty Total Market (Nifty 500 + Microcap 250).",
        [f"<b>As of</b> {as_of:%d %b %Y}", f"<b>Fetched</b> {fetched_at:%d %b, %H:%M} IST",
         f"<b>{len(result)}</b> of {len(meta)} stocks rated", f"<b>{n_liquid}</b> liquid"])
ui.kpis([
    ("Top picks", len(top), f"RS ≥ {pref_rs} + trend + RS line", "#10b981"),
    ("Trend Template", len(tt), "all 8 conditions", "#6366f1"),
    ("VCP setups", len(vcp), "tight & near pivot", "#8b5cf6"),
    ("Volume breakouts", len(brk), f"last {m.BREAKOUT_RECENT_DAYS} sessions", "#f97316"),
    ("Stage 1 bases", len(s1), "early watch-list", "#3b82f6"),
    ("RS blue dot", len(blue), "RS line leads price", "#ec4899"),
    ("Liquid stocks", n_liquid, f"of {len(result)} · ≥ ₹{m.MIN_TRADED_VALUE_CR:.0f} cr/day", "#14b8a6"),
])


def stock_cols(keys):
    """Insert the NSE basic-industry column after Sector when the classification file is loaded."""
    keys = list(keys)
    if has_cls and "Sector" in keys:
        keys.insert(keys.index("Sector") + 1, "Basic Industry")
    return keys


def show(frame, name, text, keys=None, sort=("RS_Rating", -1), titles=None):
    ui.note(f"<b>{len(frame)}</b> stocks · {text}")
    ui.table(frame, stock_cols(keys or ui.COLS_DEFAULT), sort, font, density, titles=titles)
    if not frame.empty:
        st.download_button("⬇️ Download CSV", frame.to_csv(index=False).encode("utf-8"),
                           file_name=f"totalmarket_{name}_{as_of.date()}.csv", mime="text/csv",
                           key=f"dl_{name}")


def show_breadth_and_sectors():
    b = m.market_breadth(result)
    ui.note(f"Every figure on this tab uses <b>liquid stocks only</b> ({LIQ_TEXT}): "
            f"<b>{b['stocks']}</b> of {len(result)} rated stocks. Sidebar filters don't apply here.")
    ui.adv_dec(b["advancers"], b["decliners"], b["unchanged"])
    ui.kpis([
        ("52-week highs", b["new_52w_highs"], "closing basis", "#10b981"),
        ("52-week lows", b["new_52w_lows"], "closing basis", "#ef4444"),
        ("RS line at high", f"{b['pct_rsline_high']:.0f}%", "of liquid stocks", "#ec4899"),
        ("RS ≥ 80", f"{b['pct_rs80']:.0f}%", "market leaders", "#6366f1"),
    ])
    ui.meters([
        ("Above 50-day SMA", b["pct_above_50"], "short-term trend"),
        ("Above 150-day SMA", b["pct_above_150"], "intermediate trend"),
        ("Above 200-day SMA", b["pct_above_200"], "long-term trend"),
        ("Stage 2 · Advancing", b["pct_stage2"], "price above a rising 30-week average"),
        ("Pass Trend Template", b["pct_tt"], "all 8 Minervini conditions"),
    ])

    st.markdown("#### Sector & industry strength")
    levels = ["Sector"] + (["Industry", "Basic Industry"] if has_cls else [])
    level = st.radio("Group by", levels, horizontal=True,
                     help="Sector = NSE's 22 sectors (from NSE's index list). Industry and Basic "
                          "Industry come from NSE's 4-level classification file.")
    if not has_cls:
        st.caption("Industry / basic-industry levels need `nse_classification.csv` next to app.py - see "
                   "the notes at the bottom of the page for the one-time setup.")
    min_n = st.slider("Minimum liquid stocks per group", 1, 10, 3,
                      help="Small groups give noisy scores, so they are hidden below this size.")
    ind = m.industry_strength(result, min_stocks=min_n, level=level)
    if ind.empty:
        st.info("No groups meet the minimum size.")
        return
    ui.note(f"Ranked by a blend of <b>median RS Rating (50%)</b>, <b>share of stocks with RS ≥ 80 (25%)</b> "
            f"and <b>share passing the Trend Template (25%)</b>.")
    ui.table(ind, ui.COLS_INDUSTRY, ("Rank", 1), font, density, titles={"Group": level})
    st.download_button("⬇️ Download CSV", ind.to_csv(index=False).encode("utf-8"),
                       file_name=f"totalmarket_{level.lower().replace(' ', '_')}_strength_{as_of.date()}.csv",
                       mime="text/csv", key="dl_group_table")
    st.markdown(f"#### Stocks in a {level.lower()}")
    pick = st.selectbox(f"Choose a {level.lower()} (listed by strength)", ind["Group"].tolist())
    members = result[(result[level] == pick) & result["Liquid"]].sort_values("RS_Rating", ascending=False)
    show(members, "group_stocks", f"Liquid stocks in <b>{pick}</b>, sorted by RS Rating.")


(tab_top, tab_tt, tab_vcp, tab_brk, tab_s1, tab_blue, tab_mkt, tab_all) = st.tabs([
    "Top picks", "Trend Template", "VCP setups", "Breakouts", "Stage 1 bases", "RS blue dot",
    "Breadth & sectors", "All stocks"])

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
with tab_mkt:
    show_breadth_and_sectors()
with tab_all:
    show(df.sort_values("RS_Rating", ascending=False), "all",
         "Every rated stock" + (" (liquid only)" if liquid_only else "") +
         (f" with market cap above ₹{min_cap:,} cr" if cap_ok and min_cap > 0 else "") +
         ". <b>Click any column header to sort.</b>")

with st.expander("Notes & limitations"):
    st.markdown(
        f"- **Universe:** the Nifty Total Market list from NSE (Nifty 500 + Nifty Microcap 250, about 750 stocks). "
        f"Placeholder `DUMMY…` rows are ignored; a few REIT units are included as NSE lists them.\n"
        f"- **RS Rating** ranks every rated stock in this universe (1-99). Stocks with less than 12 months of "
        f"price history, or with no Yahoo Finance data, are not rated.\n"
        f"- **Liquid** = {LIQ_TEXT} (close × volume). The *Liquid stocks only* switch filters the stock lists; "
        f"breadth, sector and industry figures always use liquid stocks.\n"
        f"- **Sectors** are NSE's 22 sectors, taken from NSE's index list. **Industry** and **Basic industry** come "
        f"from NSE's 4-level classification and appear only when `nse_classification.csv` is present "
        f"(currently: {'loaded' if has_cls else 'not loaded'}).\n"
        f"- **Market cap** = shares outstanding (Yahoo Finance, cached for a week) × the latest close, in ₹ crore. "
        f"Yahoo's share counts are occasionally stale or off for small companies, so treat borderline cases with care.\n"
        f"- 52-week high/low use closing prices, not intraday extremes.\n"
        f"- **Stages** use the 150-day SMA (about the 30-week average): Stage 2 = price above a rising average, "
        f"Stage 4 = below a falling one; Stage 1/3 = flat average after a decline/advance.\n"
        f"- **VCP, bases and breakouts are rule-based approximations** computed on daily closes/highs/lows. "
        f"They will flag some patterns a human would reject and miss others, so always confirm on the chart.\n"
        f"- RS Line is measured against the Nifty 500 index (Yahoo Finance); Yahoo data can have gaps.\n"
        f"- Educational tool, not investment advice."
    )
    if not has_cls:
        st.markdown(
            "**One-time setup for industry & basic-industry levels:** in Colab or on your PC run "
            "`build_classification_file()` from `nse500_rs_rating.py` (it asks NSE for each stock's "
            "classification, about 8-10 minutes), then add the saved `nse_classification.csv` to the "
            "GitHub repository next to `app.py`.")
