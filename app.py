"""
NSE 500 RS Rating + Minervini Trend Template scanner - Streamlit app.

Run locally:   pip install -r requirements.txt && streamlit run app.py
Deploy free:   push this folder to GitHub -> share.streamlit.io -> New app -> app.py
Requires nse500_rs_rating.py in the same folder (holds all the calculations).
"""
import io
import os
import sys
import urllib.parse
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nse500_rs_rating as m  # noqa: E402

CACHE_HOURS = 6
IST = timezone(timedelta(hours=5, minutes=30))

st.set_page_config(page_title="NSE 500 RS Scanner", page_icon="📈", layout="wide")


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
    prices = m.download_prices(meta["Ticker"].tolist())
    bench = m.get_benchmark()
    result, as_of = m.build_table(prices, bench, meta)
    return result, as_of, datetime.now(IST), prices.shape[1]


# ---------------- Header ----------------
st.title("📈 NSE 500 RS Rating Scanner")
st.caption("RS Rating (1-99) = percentile rank of the weighted 3/6/9/12-month return (40/20/20/20) "
           "within the NSE 500, plus Minervini Trend Template checks. Data: Yahoo Finance.")

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
    st.header("Filters")
    min_rs = st.slider("Minimum RS Rating", 1, 99, 1)
    pref_rs = st.slider("RS for 'Top picks'", 50, 99, m.PREFERRED_RS)
    rsi_min = st.slider("RSI above (Top picks)", 0, 80, 50)
    industries = sorted(result["Industry"].dropna().unique())
    pick_ind = st.multiselect("Industry", industries)
    q = st.text_input("Search symbol / company")
    st.divider()
    st.caption(f"Prices as of **{as_of.date()}**  \nFetched {fetched_at:%d %b %Y, %H:%M} IST  \n"
               f"Auto-refreshes every {CACHE_HOURS} hours")
    if st.button("🔄 Refresh data now"):
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

c1, c2, c3, c4 = st.columns(4)
c1.metric("Stocks rated", len(result))
c2.metric("Top picks", len(top))
c3.metric("Trend Template pass", len(tt))
c4.metric("RS blue dot", len(blue))

COLS = ["Symbol", "Company Name", "Industry", "RS_Rating", "RS_Change", "TT_Score",
        "RS_Line_New_High", "RS_Blue_Dot", "RSI14", "Close", "Pct_From_52W_High",
        "Pct_Above_52W_Low", "Weighted_Return_%", "Chart"]
CFG = {
    "Symbol": st.column_config.TextColumn("Symbol"),
    "Company Name": st.column_config.TextColumn("Company"),
    "RS_Rating": st.column_config.ProgressColumn("RS", min_value=0, max_value=99, format="%d"),
    "RS_Change": st.column_config.NumberColumn("Chg", format="%d"),
    "TT_Score": st.column_config.NumberColumn("TT score (of 8)", format="%d"),
    "RS_Line_New_High": st.column_config.CheckboxColumn("RS line high"),
    "RS_Blue_Dot": st.column_config.CheckboxColumn("Blue dot"),
    "RSI14": st.column_config.NumberColumn("RSI", format="%.1f"),
    "Close": st.column_config.NumberColumn("Close", format="%.2f"),
    "Pct_From_52W_High": st.column_config.NumberColumn("% from 52W high", format="%.1f"),
    "Pct_Above_52W_Low": st.column_config.NumberColumn("% above 52W low", format="%.1f"),
    "Weighted_Return_%": st.column_config.NumberColumn("Weighted return %", format="%.1f"),
    "Chart": st.column_config.LinkColumn("Chart", display_text="TradingView"),
}


def show(frame: pd.DataFrame, name: str, note: str):
    st.caption(note)
    if frame.empty:
        st.info("No stocks match the current filters.")
        return
    view = frame.copy()
    view["Chart"] = "https://www.tradingview.com/chart/?symbol=NSE%3A" + \
        view["Symbol"].map(lambda s: urllib.parse.quote(s, safe=""))
    st.dataframe(view[COLS], hide_index=True, column_config=CFG, height=600)
    st.download_button("⬇️ Download CSV", frame.to_csv(index=False).encode("utf-8"),
                       file_name=f"nse500_{name}_{as_of.date()}.csv", mime="text/csv",
                       key=f"dl_{name}")


IND_CFG = {
    "Rank": st.column_config.NumberColumn("Rank", format="%d"),
    "Industry_Score": st.column_config.ProgressColumn("Strength (1-99)", min_value=0, max_value=99, format="%d"),
    "Stocks": st.column_config.NumberColumn("Stocks", format="%d"),
    "Median_RS": st.column_config.NumberColumn("Median RS", format="%.0f"),
    "Avg_RS": st.column_config.NumberColumn("Avg RS", format="%.1f"),
    "Avg_RS_Change": st.column_config.NumberColumn("Avg RS chg", format="%.1f"),
    "Pct_RS80": st.column_config.NumberColumn("% stocks RS>=80", format="%.0f"),
    "Pct_TT_Pass": st.column_config.NumberColumn("% pass Trend Template", format="%.0f"),
    "TT_Pass_Count": st.column_config.NumberColumn("TT pass #", format="%d"),
    "Blue_Dots": st.column_config.NumberColumn("Blue dots", format="%d"),
    "Avg_Weighted_Return_%": st.column_config.NumberColumn("Avg weighted return %", format="%.1f"),
    "Top_Stocks": st.column_config.TextColumn("Top 3 stocks (by RS)"),
}


def show_industries():
    st.caption("Industries ranked by a blend of median RS Rating (50%), share of stocks with RS >= 80 (25%) "
               "and share passing the Trend Template (25%). Uses the whole NSE 500 (sidebar filters don't apply).")
    min_n = st.slider("Minimum stocks per industry", 1, 10, 3,
                      help="Small industries give noisy scores, so they are hidden below this size.")
    ind = m.industry_strength(result, min_stocks=min_n)
    if ind.empty:
        st.info("No industries meet the minimum size.")
        return
    st.dataframe(ind, hide_index=True, column_config=IND_CFG, height=520)
    st.download_button("⬇️ Download CSV", ind.to_csv(index=False).encode("utf-8"),
                       file_name=f"nse500_industry_strength_{as_of.date()}.csv", mime="text/csv",
                       key="dl_industry_table")
    st.subheader("Stocks in an industry")
    pick = st.selectbox("Choose an industry (listed by strength)", ind["Industry"].tolist())
    members = result[result["Industry"] == pick].sort_values("RS_Rating", ascending=False)
    show(members, "industry_stocks", f"All rated stocks in {pick}, sorted by RS Rating.")


tab1, tab2, tab3, tab4, tab5 = st.tabs([
    f"Top picks ({len(top)})", f"Trend Template pass ({len(tt)})", f"RS blue dot ({len(blue)})",
    "Industry strength", f"All stocks ({len(df)})"])
with tab1:
    show(top, "top_picks", f"Passes all 8 Trend Template conditions, RS >= {pref_rs}, "
                           f"RS line within 5% of its 52-week high, RSI > {rsi_min}.")
with tab2:
    show(tt, "trend_template", "All 8 Minervini conditions met (price vs 50/150/200 SMA, "
                               "200 SMA rising, 30% above 52W low, within 25% of 52W high, RS >= 70).")
with tab3:
    show(blue, "blue_dot", f"RS line at a 52-week high while price is still >2% below its own high "
                           f"(RS >= {pref_rs}).")
with tab4:
    show_industries()
with tab5:
    show(df.sort_values("RS_Rating", ascending=False), "all", "Every rated stock, sorted by RS Rating. "
                                                             "Click any column header to re-sort.")

with st.expander("Notes & limitations"):
    st.markdown(
        "- Stocks with less than 12 months of price history are not rated.\n"
        "- 52-week high/low use closing prices, not intraday extremes.\n"
        "- RS Rating is relative to the NSE 500 only, so it can differ from screeners that rank a wider universe.\n"
        "- Yahoo Finance data can have gaps; a few tickers may be missing on any given day.\n"
        "- Educational tool, not investment advice."
    )
