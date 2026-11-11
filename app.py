"""
Nifty Total Market RS Rating + Minervini Trend Template + VCP / breakout / Stage scanner - Streamlit app.

Run locally:   pip install -r requirements.txt && streamlit run app.py
Deploy free:   push this folder to GitHub -> share.streamlit.io -> New app -> app.py
Files needed in the same folder: app.py, ui.py, nse500_rs_rating.py, requirements.txt
"""
import io
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nse500_rs_rating as m  # noqa: E402
import ui  # noqa: E402

CACHE_HOURS = 6
IST = timezone(timedelta(hours=5, minutes=30))

st.set_page_config(page_title="Nifty Total Market RS Scanner", page_icon="📈", layout="wide")

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
    industry_col = next((c for c in ("Basic Industry", "Basic_Industry")
                         if c in df.columns), None)
    if industry_col is None and "Industry" in df.columns:
        industry_col = "Industry"
    if industry_col is None:
        raise ValueError("Constituent file must include an Industry or Basic Industry column")
    out = df[["Company Name", "Symbol"]].copy()
    out["Basic Industry"] = df[industry_col].astype("string").str.strip()
    # The current constituent feed labels its basic-industry field `Industry`.
    if "Industry" in df.columns:
        out["Industry"] = df["Industry"].astype("string").str.strip()
    else:
        out["Industry"] = out["Basic Industry"]
    if "Sector" in df.columns:
        out["Sector"] = df["Sector"].astype("string").str.strip()
    df = out
    df["Ticker"] = df["Symbol"].str.strip() + ".NS"
    return df


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def fetch_universe():
    """Nifty Total Market list: official constituent download -> local CSV -> None."""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    for url in m.NIFTY_TOTAL_MARKET_URLS:
        try:
            r = requests.get(url, timeout=30, headers=headers)
            r.raise_for_status()
            return parse_universe(pd.read_csv(io.StringIO(r.text)))
        except Exception:
            continue
    local = os.path.join(HERE, "ind_niftytotalmarket_list.csv")
    if os.path.exists(local):
        return parse_universe(pd.read_csv(local))
    return None


@st.cache_data(ttl=CACHE_HOURS * 3600, show_spinner=False)
def compute(meta: pd.DataFrame):
    md = m.download_market_data(meta["Ticker"].tolist())
    prices = md["Close"]
    bench = m.get_benchmark()
    result, as_of = m.build_table(prices, bench, meta, ohlcv=md)
    return result, as_of, datetime.now(IST), prices.shape[1], md, bench


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def fetch_market_caps(tickers: tuple[str, ...]) -> dict[str, float]:
    """Fetch Yahoo Finance market caps in ₹ crore; missing values are omitted."""
    def fetch_one(ticker: str):
        try:
            value = yf.Ticker(ticker).fast_info["market_cap"]
            value = float(value)
            return ticker, value / 10_000_000 if value > 0 else None
        except Exception:
            return ticker, None

    with ThreadPoolExecutor(max_workers=12) as executor:
        pairs = executor.map(fetch_one, tickers)
        return {ticker: value for ticker, value in pairs if value is not None}


meta = fetch_universe()
if meta is None:
    st.warning("Could not download the Nifty Total Market constituent list. Upload "
               "`ind_niftytotalmarket_list.csv` from Nifty Indices.")
    up = st.file_uploader("ind_niftytotalmarket_list.csv", type="csv")
    if up is None:
        st.stop()
    meta = parse_universe(pd.read_csv(up))

try:
    with st.spinner("Downloading Nifty Total Market stocks from Yahoo Finance (first load may take several minutes)..."):
        result, as_of, fetched_at, n_prices, market_data, bench = compute(meta)
        prices = market_data["Close"]
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
    industries = sorted(result["Basic Industry"].dropna().unique())
    pick_ind = st.multiselect("Basic industry", industries)
    liquid_only = st.checkbox("Only show liquid stocks", value=False,
                              help="Liquid means average close × volume of at least ₹10 crore over the last 20 sessions.")
    min_market_cap = st.selectbox("Minimum market cap", [0, 500, 1000], format_func=lambda v: {
        0: "Any market cap", 500: "> ₹500 Cr", 1000: "> ₹1,000 Cr"
    }[v], help="Yahoo Finance market cap, cached for 24 hours. Fetching it for the full universe happens only when a cap filter is selected.")
    q = st.text_input("Search symbol / company")
    st.divider()
    st.caption(f"Prices as of **{as_of.date()}**  \nFetched {fetched_at:%d %b %Y, %H:%M} IST  \n"
               f"Auto-refreshes every {CACHE_HOURS} hours")
    if st.button("🔄 Refresh data now", use_container_width=True):
        compute.clear()
        st.rerun()

# ---------------- Apply filters ----------------
if min_market_cap:
    cap_tickers = tuple(sorted({f"{str(s).strip()}.NS" for s in result["Symbol"].dropna()}))
    with st.spinner("Loading market caps for the Nifty Total Market stocks…"):
        market_caps = fetch_market_caps(cap_tickers)
    result["Market_Cap_Cr"] = result["Symbol"].map(
        lambda s: market_caps.get(f"{str(s).strip()}.NS") if pd.notna(s) else None)
    cap_coverage = int(result["Market_Cap_Cr"].notna().sum())
    if cap_coverage < len(result):
        st.warning(f"Market cap data is available for {cap_coverage} of {len(result)} rated stocks. Stocks without a value are excluded by the active cap filter.")

df = result[result["RS_Rating"] >= min_rs]
if pick_ind:
    df = df[df["Basic Industry"].isin(pick_ind)]
if liquid_only:
    df = df[df["Liquid"].fillna(False)]
if min_market_cap:
    df = df[df["Market_Cap_Cr"] > min_market_cap]
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
        "Weinstein stages across the Nifty Total Market universe.",
        [f"<b>As of</b> {as_of:%d %b %Y}", f"<b>Fetched</b> {fetched_at:%d %b, %H:%M} IST",
         f"<b>{len(result)}</b> stocks rated",
         f"<b>{int(result['Liquid'].fillna(False).sum())}</b> liquid ≥ ₹10Cr ADTV"])
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
                           file_name=f"nifty_total_market_{name}_{as_of.date()}.csv", mime="text/csv",
                           key=f"dl_{name}")


def show_industries():
    ui.note("Basic industries ranked by a blend of <b>median RS Rating (50%)</b>, <b>share of liquid stocks with RS ≥ 80 "
            "(25%)</b> and <b>share passing the Trend Template (25%)</b>. Only stocks averaging at least "
            "₹10 crore of close × volume over the last 20 sessions are included; sidebar filters don't apply.")
    min_n = st.slider("Minimum stocks per industry", 1, 10, 3,
                      help="Small industries give noisy scores, so they are hidden below this size.")
    liquid_result = result[result["Liquid"].fillna(False)]
    ind = m.industry_strength(liquid_result, min_stocks=min_n)
    if ind.empty:
        st.info("No industries meet the minimum size.")
        return
    ui.table(ind, ui.COLS_INDUSTRY, ("Rank", 1), font, density)
    st.download_button("⬇️ Download CSV", ind.to_csv(index=False).encode("utf-8"),
                       file_name=f"nifty_total_market_basic_industry_strength_{as_of.date()}.csv", mime="text/csv",
                       key="dl_industry_table")
    st.markdown("#### Stocks in an industry")
    pick = st.selectbox("Choose a basic industry (listed by strength)", ind["Basic Industry"].tolist())
    members = liquid_result[liquid_result["Basic Industry"] == pick].sort_values("RS_Rating", ascending=False)
    show(members, "basic_industry_stocks", f"Liquid rated stocks in <b>{pick}</b>, sorted by RS Rating.")


def show_breadth():
    liquid = result[result["Liquid"].fillna(False)].copy()
    if liquid.empty:
        st.info("No stocks meet the liquidity threshold with 20 complete sessions of data.")
        return
    metric_map = {
        "Above 50-day SMA": liquid["Close"] > liquid["SMA50"],
        "Above 150-day SMA": liquid["Close"] > liquid["SMA150"],
        "Above 200-day SMA": liquid["Close"] > liquid["SMA200"],
        "Trend Template pass": liquid["TT_Pass"],
        "Stage 2 advancing": liquid["Stage"].astype("string").str.startswith("Stage 2", na=False),
    }
    st.markdown("#### Nifty Total Market breadth")
    st.caption("Breadth uses liquid stocks only: 20-session average close × volume ≥ ₹10 crore.")
    cols = st.columns(len(metric_map) + 1)
    cols[0].metric("Liquid stocks", f"{len(liquid)} / {len(result)}")
    for col, (label, condition) in zip(cols[1:], metric_map.items()):
        pct = condition.fillna(False).mean() * 100
        col.metric(label, f"{pct:.1f}%", f"{int(condition.fillna(False).sum())} stocks")
    summaries = []
    for group_col in ("Sector", "Industry", "Basic Industry"):
        if group_col not in liquid.columns or not liquid[group_col].notna().any():
            continue
        if group_col != "Basic Industry" and liquid[group_col].equals(liquid["Basic Industry"]):
            continue
        rows = []
        for group, frame in liquid.groupby(group_col, dropna=True):
            row = {group_col: group, "Liquid Stocks": len(frame)}
            for label, source in [("Above SMA50 %", "SMA50"), ("Above SMA150 %", "SMA150"),
                                  ("Above SMA200 %", "SMA200")]:
                valid = frame["Close"].notna() & frame[source].notna()
                row[label] = round((frame.loc[valid, "Close"] > frame.loc[valid, source]).mean() * 100, 1) if valid.any() else None
            row["Trend Template %"] = round(frame["TT_Pass"].fillna(False).mean() * 100, 1)
            row["Avg 20D Traded Value (₹Cr)"] = round(frame["Avg_Traded_Value_20D_Cr"].mean(), 2)
            rows.append(row)
        if rows:
            summary = pd.DataFrame(rows).sort_values("Liquid Stocks", ascending=False)
            summaries.append((group_col, summary))
    if not summaries:
        st.info("No sector, industry or basic-industry labels are available for the liquid stocks.")
        return
    for group_col, summary in summaries:
        st.markdown(f"#### {group_col} breadth (liquid stocks)")
        st.dataframe(summary, hide_index=True, use_container_width=True)
    csv_summary = summaries[-1][1]
    st.download_button("⬇️ Download breadth CSV", csv_summary.to_csv(index=False).encode("utf-8"),
                       file_name=f"nifty_total_market_breadth_{as_of.date()}.csv", mime="text/csv",
                       key="dl_liquid_breadth")


def detect_base_history(bars: pd.DataFrame) -> list[dict]:
    """Find distinct base episodes in roughly the last 12 months for the stock chart."""
    data = bars.tail(252).copy()
    if len(data) < min(m.BASE_WINDOWS):
        return []
    close = data["Close"].to_numpy(float)
    high = data["High"].to_numpy(float)
    low = data["Low"].to_numpy(float)
    volume = data["Volume"].to_numpy(float)
    avg_volume = bars["Volume"].shift(1).rolling(50).mean().reindex(data.index).to_numpy(float)
    sma50 = bars["Close"].rolling(50).mean().reindex(data.index).to_numpy(float)
    dates = data.index
    found: list[dict] = []
    cursor = 0
    n = len(data)

    while cursor + min(m.BASE_WINDOWS) <= n:
        candidate = None
        for end in range(cursor + min(m.BASE_WINDOWS) - 1, n):
            for window in sorted(m.BASE_WINDOWS, reverse=True):
                start = end - window + 1
                if start < cursor:
                    continue
                segment_high = float(high[start:end + 1].max())
                segment_low = float(low[start:end + 1].min())
                depth = (segment_high - segment_low) / segment_high if segment_high else 1.0
                drift = close[end] / close[start] - 1 if close[start] else 1.0
                if depth <= m.MAX_BASE_DEPTH and abs(drift) <= m.MAX_BASE_DRIFT:
                    candidate = {"start": start, "end": end, "pivot": segment_high,
                                 "low": segment_low, "depth": depth * 100,
                                 "window": window}
                    break
            if candidate:
                break
        if candidate is None:
            break

        start = candidate["start"]
        base_end = candidate["end"]
        pivot = candidate["pivot"]
        base_low = candidate["low"]
        breakout_at = None
        failed_before_breakout = False
        scan_end = min(n - 1, start + max(m.BASE_WINDOWS))
        for j in range(candidate["end"] + 1, scan_end + 1):
            if close[j] > pivot:
                breakout_at = j
                base_end = j - 1
                break
            base_end = j
            pivot = max(pivot, float(high[j]))
            base_low = min(base_low, float(low[j]))
            if close[j] < base_low * 0.92:
                failed_before_breakout = True
                break

        if breakout_at is not None:
            post = np.arange(breakout_at, n)
            failure = next((int(j) for j in post
                            if close[j] < pivot * 0.92 or
                            (np.isfinite(sma50[j]) and close[j] < sma50[j])), None)
            outcome_end = failure if failure is not None else n - 1
            best_pct = (float(high[breakout_at:outcome_end + 1].max()) / pivot - 1) * 100
            now_pct = (float(close[n - 1]) / pivot - 1) * 100
            vol_x = (volume[breakout_at] / avg_volume[breakout_at]
                     if np.isfinite(avg_volume[breakout_at]) and avg_volume[breakout_at] > 0 else np.nan)
            if failure is not None:
                status = f"Failed {dates[failure]:%d %b %y} · best +{best_pct:.1f}% first"
            else:
                status = f"Holding · best +{best_pct:.1f}% · now {now_pct:+.1f}%"
            cursor = breakout_at + 1
        elif failed_before_breakout:
            vol_x = np.nan
            status = "Failed · no breakout"
            cursor = base_end + 1
        else:
            vol_x = np.nan
            under_pivot = (1 - close[n - 1] / pivot) * 100
            status = f"Forming · {under_pivot:.1f}% under pivot"
            cursor = n

        base_days = base_end - start + 1
        box = data.iloc[start:base_end + 1]
        found.append({
            "Started": dates[start], "Length": f"{max(1, round(base_days / 5))} wk",
            "Depth": f"{(1 - float(box['Low'].min()) / float(box['High'].max())) * 100:.0f}%",
            "Pivot": float(box["High"].max()),
            "Breakout": dates[breakout_at] if breakout_at is not None else None,
            "Volume": f"{vol_x:.1f}×" if np.isfinite(vol_x) else "—",
            "Then": status, "_Start": dates[start], "_End": dates[base_end],
            "_Low": float(box["Low"].min()), "_High": float(box["High"].max()),
        })
        if cursor <= candidate["end"]:
            cursor = candidate["end"] + 1
    return found


(tab_top, tab_tt, tab_vcp, tab_brk, tab_s1, tab_blue, tab_ind, tab_all, tab_detail, tab_breadth) = st.tabs([
    f"Top picks · {len(top)}", f"Trend Template · {len(tt)}", f"VCP setups · {len(vcp)}",
    f"Breakouts · {len(brk)}", f"Stage 1 bases · {len(s1)}", f"RS blue dot · {len(blue)}",
    "Industries", f"All stocks · {len(df)}", "Stock detail", "Market breadth"])

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

with tab_detail:
    st.markdown("### Stock detail")
    st.caption("Explore recent price action, moving averages and relative performance for any rated stock.")
    choices = result.sort_values("RS_Rating", ascending=False)["Symbol"].dropna().tolist()
    if not choices:
        st.info("No rated stocks are available for the detail view.")
    else:
        default_symbol = st.session_state.get("detail_symbol", choices[0])
        if default_symbol not in choices:
            default_symbol = choices[0]
        symbol = st.selectbox("Choose a stock", choices,
                              index=choices.index(default_symbol), key="detail_symbol")
        row = result.loc[result["Symbol"] == symbol].iloc[0]
        # build_table removes its internal Ticker join key from the display result.
        ticker = f"{str(row['Symbol']).strip()}.NS"
        if ticker not in prices.columns:
            st.warning("Price history is unavailable for this stock in the current scan.")
        else:
            history_days = {"3 months": 63, "6 months": 126, "1 year": 252, "2 years": 504}
            period = st.radio("Chart period", list(history_days), index=2, horizontal=True,
                              key="detail_period")
            show_base = st.checkbox("Draw detected base", value=True, key="detail_show_base")
            show_pivot = st.checkbox("Show pivot and 52-week high", value=True,
                                     key="detail_show_pivot")
            close_all = prices[ticker].dropna()
            bars = pd.DataFrame({
                "Open": market_data["Open"][ticker].reindex(close_all.index),
                "High": market_data["High"][ticker].reindex(close_all.index),
                "Low": market_data["Low"][ticker].reindex(close_all.index),
                "Close": close_all,
                "Volume": market_data["Volume"][ticker].reindex(close_all.index),
            }).dropna(subset=["Open", "High", "Low", "Close"])
            bars["EMA 10"] = bars["Close"].ewm(span=10, adjust=False).mean()
            bars["EMA 21"] = bars["Close"].ewm(span=21, adjust=False).mean()
            bars["SMA 50"] = bars["Close"].rolling(50).mean()
            bars["SMA 200"] = bars["Close"].rolling(200).mean()
            bars["Volume SMA 50"] = bars["Volume"].rolling(50).mean()
            bars["RS line"] = bars["Close"].div(bench.reindex(bars.index).ffill())
            bars["RS line"] = bars["RS line"].div(bars["RS line"].dropna().iloc[0]).mul(100)
            base_history = detect_base_history(bars)
            chart = bars.tail(history_days[period]).copy()
            rs_visible = chart["RS line"].dropna()
            if not rs_visible.empty:
                chart.loc[:, "RS line"] = chart["RS line"].div(rs_visible.iloc[0]).mul(100)

            fig = make_subplots(
                rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.035,
                row_heights=[0.62, 0.18, 0.20],
                subplot_titles=(f"{symbol} · daily candles", "Volume", "RS line vs Nifty 500 · rebased to 100"),
            )
            fig.add_trace(go.Candlestick(
                x=chart.index, open=chart["Open"], high=chart["High"],
                low=chart["Low"], close=chart["Close"], name="OHLC",
                increasing_line_color="#12805c", decreasing_line_color="#c74440",
                increasing_fillcolor="#12805c", decreasing_fillcolor="#c74440",
            ), row=1, col=1)
            for col, color in (("EMA 10", "#202c25"), ("EMA 21", "#f59e0b"),
                               ("SMA 50", "#3478c8"), ("SMA 200", "#d33b3b")):
                fig.add_trace(go.Scatter(x=chart.index, y=chart[col], name=col,
                                         mode="lines", line=dict(color=color, width=1.5)),
                              row=1, col=1)
            volume_colors = ["#12805c" if c >= o else "#c74440"
                             for o, c in zip(chart["Open"], chart["Close"])]
            fig.add_trace(go.Bar(x=chart.index, y=chart["Volume"], name="Volume",
                                 marker_color=volume_colors, opacity=0.72), row=2, col=1)
            fig.add_trace(go.Scatter(x=chart.index, y=chart["Volume SMA 50"],
                                     name="Volume SMA 50", mode="lines",
                                     line=dict(color="#697386", width=1, dash="dot")), row=2, col=1)
            fig.add_trace(go.Scatter(x=chart.index, y=chart["RS line"], name="RS line",
                                     mode="lines", line=dict(color="#16834c", width=2)), row=3, col=1)

            if show_base:
                for base_number, base in enumerate(base_history, start=1):
                    if base["_End"] < chart.index[0] or base["_Start"] > chart.index[-1]:
                        continue
                    x0 = max(base["_Start"], chart.index[0])
                    x1 = min(base["_End"], chart.index[-1])
                    fig.add_shape(type="rect", xref="x", yref="y",
                                  x0=x0, x1=x1, y0=base["_Low"], y1=base["_High"],
                                  line=dict(color="#3487d8", width=1.5, dash="dash"),
                                  fillcolor="rgba(52,135,216,0.10)")
                    fig.add_annotation(x=x0, y=base["_High"], xref="x", yref="y",
                                       text=f"B{base_number}", showarrow=False,
                                       xanchor="left", yanchor="bottom",
                                       font=dict(size=10, color="#2467a8"),
                                       bgcolor="rgba(255,255,255,0.78)")
            if show_pivot:
                pivot = row.get("Pivot")
                if pd.notna(pivot):
                    fig.add_hline(y=float(pivot), row=1, col=1,
                                  line_color="#8b5cf6", line_dash="dash", line_width=1.2,
                                  annotation_text=f"Pivot ₹{float(pivot):.2f}",
                                  annotation_position="top left")
                high_52 = close_all.tail(252).max()
                if pd.notna(high_52):
                    fig.add_hline(y=float(high_52), row=1, col=1,
                                  line_color="#b97725", line_dash="dot", line_width=1,
                                  annotation_text=f"52W high ₹{float(high_52):.2f}",
                                  annotation_position="top right")
            fig.update_layout(
                height=780, template="plotly_white", hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02,
                            xanchor="left", x=0),
                margin=dict(l=12, r=16, t=70, b=12),
                xaxis_rangeslider_visible=False,
            )
            fig.update_xaxes(rangeslider_visible=False, showgrid=True, gridcolor="#e7eaf0")
            fig.update_yaxes(title_text="Price (₹)", row=1, col=1, gridcolor="#e7eaf0")
            fig.update_yaxes(title_text="Shares", row=2, col=1, gridcolor="#e7eaf0")
            fig.update_yaxes(title_text="RS (100)", row=3, col=1, gridcolor="#e7eaf0")
            st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
            st.markdown("### Base history · last 12 months · newest first")
            if base_history:
                history_rows = []
                for base_number, base in reversed(list(enumerate(base_history, start=1))):
                    history_rows.append({
                        "Base": f"B{base_number}",
                        "Started": base["Started"].strftime("%d %b %y"),
                        "Length": base["Length"], "Depth": base["Depth"],
                        "Pivot": round(base["Pivot"], 2),
                        "Breakout": base["Breakout"].strftime("%d %b %y") if base["Breakout"] is not None else "—",
                        "Volume": base["Volume"], "Then": base["Then"],
                    })
                st.dataframe(pd.DataFrame(history_rows), hide_index=True,
                             use_container_width=True, height=min(42 * len(history_rows) + 44, 330))
                st.caption("Failed means the stock closed 8% below the pivot or below its 50-day average after breakout. History, not a forecast.")
            else:
                st.info("No bases met the scanner's range and sideways-drift rules in the last 12 months.")
            metric_cols = st.columns(5)
            for col, label, value in zip(
                    metric_cols,
                    ["RS Rating", "Trend Template", "Stage", "RSI (14)", "Basic industry"],
                    [f"{row['RS_Rating']}", f"{row['TT_Score']}/8",
                     str(row.get("Stage", "n/a")), f"{row['RSI14']:.1f}" if pd.notna(row["RSI14"]) else "n/a",
                     str(row.get("Basic Industry", "n/a"))]):
                col.metric(label, value)
            signal_cols = [c for c in ["VCP_Setup", "Vol_Breakout", "Stage1_Base",
                                        "RS_Line_New_High", "RS_Blue_Dot"] if c in row.index]
            st.markdown("#### Current scanner signals")
            st.dataframe(pd.DataFrame({"Signal": signal_cols,
                                       "Active": ["Yes" if bool(row[c]) else "No"
                                                  for c in signal_cols]}),
                         hide_index=True, use_container_width=True)

with tab_breadth:
    show_breadth()

with st.expander("Notes & limitations"):
    st.markdown(
        "- Stocks with less than 12 months of price history are not rated.\n"
        "- 52-week high/low use closing prices, not intraday extremes.\n"
        "- RS Rating ranks the rated members of the Nifty Total Market universe.\n"
        "- Liquidity uses average daily close × volume over the latest 20 sessions; stocks below ₹10 crore are marked illiquid.\n"
        "- Industry strength and market breadth use liquid stocks only.\n"
        "- **Stages** use the 150-day SMA (about the 30-week average): Stage 2 = price above a rising average, "
        "Stage 4 = below a falling one; Stage 1/3 = flat average after a decline/advance. Stocks with under ~13 "
        "months of history show 'n/a'.\n"
        "- **VCP, bases and breakouts are rule-based approximations** computed on daily closes/highs/lows. "
        "They will flag some patterns a human would reject and miss others, so always confirm on the chart.\n"
        "- Yahoo Finance data can have gaps; a few tickers may be missing on any given day.\n"
        "- Educational tool, not investment advice."
    )
