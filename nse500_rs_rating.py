# ============================================================
# Nifty Total Market (Nifty 500 + Microcap 250) RS Rating + Minervini scanner
# Google Colab | Data: Yahoo Finance (yfinance)
#
# RS Rating (1-99):
#   Weighted Return = 40% R(3M) + 20% R(6M) + 20% R(9M) + 20% R(12M)
#   Rating = percentile rank of weighted return across the scan
#
# Minervini Trend Template (8 conditions):
#   1. Price > 150 SMA and > 200 SMA
#   2. 150 SMA > 200 SMA
#   3. 200 SMA rising for at least 1 month
#   4. 50 SMA > 150 SMA and > 200 SMA
#   5. Price > 50 SMA
#   6. Price >= 30% above 52-week low
#   7. Price within 25% of 52-week high
#   8. RS Rating >= 70
#
# Output: CSVs + a self-contained web dashboard (nse500_rs_dashboard.html)
# Extras: Weinstein stage (1-4), base detection, VCP + volume-breakout scans,
#         Liquidity flag (avg 20-day traded value >= Rs 10 crore); sector, industry and breadth
#         figures use liquid stocks only; optional NSE basic-industry classification.
#         RS Line (stock / Nifty 500) new-high flags, RS "blue dot"
#         (RS line at 52w high before price), RSI(14)
# ============================================================

# --- Cell 1: install (run once in Colab) ---
# !pip install -q yfinance

import io
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
import warnings
import requests
from urllib.parse import quote
import numpy as np
import pandas as pd
import yfinance as yf

# ---------------- Settings ----------------
WEIGHTS = {3: 0.40, 6: 0.20, 9: 0.20, 12: 0.20}   # months -> weight
BATCH_SIZE = 100
HISTORY_PERIOD = "2y"
CHANGE_DAYS = 1                  # RS change vs N trading days ago
BENCHMARKS = ["^CRSLDX", "^NSEI"]  # Nifty 500, fallback Nifty 50 (for RS Line)
TRADING_DAYS_52W = 252

MIN_RS_TT = 70                   # Trend Template condition 8
PREFERRED_RS = 80                # used for "top picks"
SMA200_RISING_DAYS = 21          # 21 = 1 month (use ~105 for the stricter 5-month test)
MIN_ABOVE_52W_LOW = 0.30         # 30% above 52w low
MAX_BELOW_52W_HIGH = 0.25        # within 25% of 52w high
RS_NEAR_HIGH = 0.95              # RS line within 5% of its 52w high = "near high"
BLUE_DOT_PRICE_GAP = 0.02        # price still >2% below its own 52w high

# ---- Universe: Nifty Total Market = Nifty 500 + Nifty Microcap 250 (about 750 stocks) ----
UNIVERSE_NAME = "Nifty Total Market"
UNIVERSE_URL = "https://nsearchives.nseindia.com/content/indices/ind_niftytotalmarket_list.csv"
UNIVERSE_FALLBACK_URLS = (       # used if the Total Market file is unavailable
    "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv",
    "https://nsearchives.nseindia.com/content/indices/ind_niftymicrocap250_list.csv",
)
LOCAL_UNIVERSE_FILE = "ind_niftytotalmarket_list.csv"   # optional local copy of the NSE list
CLASSIFICATION_FILE = "nse_classification.csv"          # optional cache: NSE industry / basic industry

# ---- Market cap (shares outstanding x latest close, in Rs crore) ----
MIN_MARKET_CAP_CR = 500.0        # stock-list filter: market cap above Rs 500 crore (0 = off)

# ---- Liquidity ----
LIQUIDITY_DAYS = 20              # sessions averaged
MIN_TRADED_VALUE_CR = 10.0       # liquid = avg daily traded value (close x volume) >= Rs 10 crore

# ---- Stage analysis (Weinstein, 30-week MA ~ 150-day SMA) ----
STAGE_SLOPE_DAYS = 30            # slope of the 150 SMA measured over 30 sessions (~6 weeks)
STAGE_PRIOR_DAYS = 80            # ...compared with the 80 sessions before that
STAGE_FLAT_SLOPE = 0.01          # |slope| <= 1% counts as "flat"
STAGE4_DIST = 0.20               # price >20% below the 150 SMA is always Stage 4

# ---- Base detection ----
BASE_WINDOWS = (30, 45, 60, 90, 120, 150)   # sessions checked (6 to 30 weeks)
MAX_BASE_DEPTH = 0.35            # base depth (high-to-low) at most 35%
MAX_BASE_DRIFT = 0.15            # price today within +/-15% of price at base start (sideways)

# ---- VCP ----
VCP_LOOKBACK = 200               # sessions searched for the left peak of the pattern
ZIGZAG_PCT = 0.04                # a pullback/rally must be >= 4% to count as a swing
MIN_CONTRACTIONS = 2             # at least 2 successively tighter pullbacks
TIGHTEN_RATIO = 0.90             # each pullback <= 90% of the previous one
VCP_LAST_MAX = 0.10              # final contraction no deeper than 10%
VOL_DRYUP_RATIO = 0.80           # last-10-day avg volume <= 80% of the 50-day avg
VCP_NEAR_PIVOT = 5.0             # "setup" = price 0-5% below the pivot

# ---- Volume breakout ----
BREAKOUT_LOOKBACK = 60           # must close above the prior 60-session high
BREAKOUT_VOL_X = 1.5             # volume >= 1.5x the prior 50-day average
BREAKOUT_RECENT_DAYS = 3         # look back over the last 3 sessions


# ---------------- 1. Universe + NSE classification ----------------
_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
_CLS_COLS = ["Macro Sector", "Industry", "Basic Industry"]


def parse_universe(df):
    """Normalise an NSE index-constituent CSV. NSE's 'Industry' column there is the NSE *Sector*
    level (22 sectors), so it becomes 'Sector'. Placeholder 'DUMMY...' rows are dropped."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"Industry": "Sector"})
    for c in ("Series", "ISIN Code"):
        if c not in df.columns:
            df[c] = ""
    df["Symbol"] = df["Symbol"].astype(str).str.strip()
    df["Sector"] = df["Sector"].astype(str).str.strip()
    dummy = (df["Symbol"].str.upper().str.startswith("DUMMY")
             | df["ISIN Code"].astype(str).str.upper().str.startswith("DUM"))
    df = df[~dummy].drop_duplicates("Symbol")
    df["Ticker"] = df["Symbol"] + ".NS"
    return df[["Company Name", "Sector", "Symbol", "Series", "ISIN Code", "Ticker"]].reset_index(drop=True)


def _download_csv(url):
    r = requests.get(url, headers=_HEADERS, timeout=30)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))


def load_universe(local_file=None):
    """Nifty Total Market constituents: official NSE CSV -> local copy -> Nifty 500 + Microcap 250.
    Returns None if nothing is available."""
    local_file = local_file or LOCAL_UNIVERSE_FILE
    try:
        return parse_universe(_download_csv(UNIVERSE_URL))
    except Exception as e:
        print(f"Could not download {UNIVERSE_NAME} list from NSE ({e}).")
    if os.path.exists(local_file):
        print(f"Using local file {local_file}")
        return parse_universe(pd.read_csv(local_file))
    try:
        parts = [_download_csv(u) for u in UNIVERSE_FALLBACK_URLS]
        print("Built the universe from Nifty 500 + Nifty Microcap 250 instead.")
        return parse_universe(pd.concat(parts, ignore_index=True))
    except Exception as e:
        print(f"Fallback lists also failed ({e}).")
        return None


def load_classification(path=None):
    """Cached NSE classification (Symbol, Macro Sector, Industry, Basic Industry); empty if absent."""
    path = path or CLASSIFICATION_FILE
    cols = ["Symbol"] + _CLS_COLS
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            if "Symbol" in df.columns:
                for c in _CLS_COLS:
                    if c not in df.columns:
                        df[c] = np.nan
                return df[cols].dropna(subset=["Symbol"]).drop_duplicates("Symbol")
        except Exception as e:
            print(f"Could not read {path} ({e})")
    return pd.DataFrame(columns=cols)


def attach_classification(meta, cls):
    """Add Macro Sector / Industry / Basic Industry (NaN where unknown) to the universe table."""
    meta = meta.drop(columns=[c for c in _CLS_COLS if c in meta.columns])
    if cls is None or cls.empty:
        for c in _CLS_COLS:
            meta[c] = np.nan
        return meta
    return meta.merge(cls[["Symbol"] + _CLS_COLS].drop_duplicates("Symbol"), on="Symbol", how="left")


def _nse_session():
    s = requests.Session()
    s.headers.update({**_HEADERS, "Accept": "application/json, text/plain, */*",
                      "Accept-Language": "en-US,en;q=0.9"})
    s.get("https://www.nseindia.com/", timeout=20)          # sets the cookies the API expects
    return s


def fetch_classification(symbols, existing=None, delay=0.6, max_consecutive_fail=15,
                         save_to=None, session=None):
    """Best-effort download of NSE's 4-level classification (macro sector, sector, industry,
    basic industry) from NSE's quote API, one symbol at a time. Resumable: symbols already in
    `existing` are skipped. NSE often blocks cloud servers, so run this on your own machine or
    Colab, then keep the saved CSV."""
    existing = existing if existing is not None else pd.DataFrame(columns=["Symbol"] + _CLS_COLS)
    done = set(existing["Symbol"]) if len(existing) else set()
    todo = [x for x in symbols if x not in done]
    if not todo:
        return existing
    sess = session or _nse_session()
    rows, fails = [], 0

    def _save():
        out = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
        if save_to:
            out.to_csv(save_to, index=False)
        return out

    for i, sym in enumerate(todo, 1):
        info = None
        try:
            r = sess.get("https://www.nseindia.com/api/quote-equity", params={"symbol": sym},
                         headers={"Referer": f"https://www.nseindia.com/get-quotes/equity?symbol={quote(sym)}"},
                         timeout=20)
            if r.status_code == 200:
                info = (r.json() or {}).get("industryInfo")
        except Exception:
            info = None
        if info and info.get("basicIndustry"):
            rows.append({"Symbol": sym, "Macro Sector": info.get("macro"), "Industry": info.get("industry"),
                         "Basic Industry": info.get("basicIndustry")})
            fails = 0
        else:
            fails += 1
            if fails >= max_consecutive_fail:
                print(f"NSE stopped answering after {i} symbols (blocked?). Saved what we have; "
                      f"run again later or from another network to continue.")
                break
        if i % 50 == 0:
            print(f"  classified {i}/{len(todo)}")
            _save()
        time.sleep(delay)
    return _save()


def build_classification_file(path=None):
    """One-off helper: download NSE industry / basic-industry for every stock in the universe
    and cache it in nse_classification.csv (run in Colab or on your PC, then upload the file
    next to app.py)."""
    path = path or CLASSIFICATION_FILE
    meta = load_universe()
    if meta is None:
        print("No universe available."); return None
    cls = fetch_classification(meta["Symbol"].tolist(), existing=load_classification(path), save_to=path)
    cov = int(cls["Basic Industry"].notna().sum()) if len(cls) else 0
    print(f"Saved {path}: {cov} of {len(meta)} stocks classified.")
    try:
        from google.colab import files
        files.download(path)
    except Exception:
        pass
    return cls


# ---------------- 2. Price download ----------------
def _clean_index(obj):
    obj.index = pd.to_datetime(obj.index)
    if obj.index.tz is not None:
        obj.index = obj.index.tz_localize(None)
    obj.index = obj.index.normalize()
    return obj


def download_market_data(tickers):
    """Download adjusted OHLCV. Returns {"Close","High","Low","Volume"} DataFrames."""
    fields = ("Close", "High", "Low", "Volume")
    frames = {k: [] for k in fields}
    for i in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[i:i + BATCH_SIZE]
        print(f"Downloading {i + 1}-{i + len(batch)} of {len(tickers)} ...")
        data = yf.download(batch, period=HISTORY_PERIOD, interval="1d",
                           auto_adjust=True, progress=False, threads=True)
        for k in fields:
            f = data[k]
            if isinstance(f, pd.Series):
                f = f.to_frame(batch[0])
            frames[k].append(f)
        time.sleep(1)

    panels = {}
    for k in fields:
        df = pd.concat(frames[k], axis=1)
        df = df.loc[:, ~df.columns.duplicated()]
        panels[k] = _clean_index(df.sort_index())

    close = panels["Close"].dropna(axis=1, how="all")
    # drop stale tickers (delisted / merged / suspended)
    last_valid = close.apply(lambda s: s.last_valid_index())
    stale = last_valid < close.index[-1] - pd.Timedelta(days=10)
    if stale.any():
        print(f"Dropping {stale.sum()} stale tickers: {list(close.columns[stale])[:10]}")
    keep = close.columns[~stale]

    out = {"Close": close[keep].ffill()}
    for k in ("High", "Low"):
        out[k] = panels[k].reindex(index=close.index, columns=keep).ffill()
    out["Volume"] = panels["Volume"].reindex(index=close.index, columns=keep)
    return out


def download_prices(tickers):
    """Close prices only (kept for backward compatibility)."""
    return download_market_data(tickers)["Close"]


def get_benchmark():
    for sym in BENCHMARKS:
        try:
            d = yf.download(sym, period=HISTORY_PERIOD, interval="1d",
                            auto_adjust=True, progress=False)
            s = d["Close"]
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
            s = s.dropna()
            if len(s) > 250:
                print(f"RS Line benchmark: {sym}")
                return _clean_index(s)
        except Exception as e:
            print(f"Benchmark {sym} failed: {e}")
    raise RuntimeError("Could not download a benchmark index from Yahoo Finance")


# ---------------- 3. RS Rating ----------------
def weighted_return(prices, end_date):
    p = prices.loc[:end_date]
    last = p.iloc[-1]
    wr = pd.Series(0.0, index=p.columns)
    for months, w in WEIGHTS.items():
        target = end_date - pd.DateOffset(months=months)
        pos = p.index.searchsorted(target, side="right") - 1
        past = pd.Series(np.nan, index=p.columns) if pos < 0 else p.iloc[pos]
        wr = wr + w * ((last / past - 1) * 100)
    return wr


def rs_rating(wr):
    wr = wr.dropna()
    pct = (wr.rank(method="min") - 1) / len(wr) * 100
    return pct.round().clip(1, 99).astype(int)


# ---------------- 4. Technicals / Trend Template ----------------
def rsi_wilder(prices, n=14):
    delta = prices.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss)


def technicals(prices, bench, rs_now):
    idx = rs_now.index
    p = prices[idx]
    last = p.iloc[-1]

    sma50 = p.rolling(50).mean()
    sma150 = p.rolling(150).mean()
    sma200 = p.rolling(200).mean()
    s50, s150, s200 = sma50.iloc[-1], sma150.iloc[-1], sma200.iloc[-1]
    s200_prev = sma200.iloc[-1 - SMA200_RISING_DAYS]

    win = p.iloc[-TRADING_DAYS_52W:]
    hi52, lo52 = win.max(), win.min()

    # --- 8 Trend Template conditions (NaN comparisons evaluate to False) ---
    c1 = (last > s150) & (last > s200)
    c2 = s150 > s200
    c3 = s200 > s200_prev
    c4 = (s50 > s150) & (s50 > s200)
    c5 = last > s50
    c6 = last >= lo52 * (1 + MIN_ABOVE_52W_LOW)
    c7 = last >= hi52 * (1 - MAX_BELOW_52W_HIGH)
    c8 = rs_now >= MIN_RS_TT
    conds = pd.concat([c1, c2, c3, c4, c5, c6, c7, c8], axis=1)
    conds.columns = ["TT1_>150&200SMA", "TT2_150>200SMA", "TT3_200SMA_up",
                     "TT4_50>150&200SMA", "TT5_>50SMA", "TT6_30%>52wLow",
                     "TT7_within25%_52wHigh", "TT8_RS>=70"]

    # --- RS Line (stock / benchmark) ---
    b = bench.reindex(p.index).ffill()
    rsl = p.div(b, axis=0)
    rsl_last = rsl.iloc[-1]
    rsl_hi = rsl.iloc[-TRADING_DAYS_52W:].max()
    rsl_new_high = rsl_last >= rsl_hi * 0.9999
    rsl_near_high = rsl_last >= rsl_hi * RS_NEAR_HIGH
    rsl_uptrend = rsl_last > rsl.rolling(50).mean().iloc[-1]
    blue_dot = rsl_new_high & (last < hi52 * (1 - BLUE_DOT_PRICE_GAP))

    out = pd.DataFrame({
        "Close": last.round(2),
        "Pct_From_52W_High": ((last / hi52 - 1) * 100).round(1),
        "Pct_Above_52W_Low": ((last / lo52 - 1) * 100).round(1),
        "SMA50": s50.round(2), "SMA150": s150.round(2), "SMA200": s200.round(2),
        "RSI14": rsi_wilder(p).iloc[-1].round(1),
        "Chg_1D_%": ((last / p.iloc[-2] - 1) * 100).round(2),
        "RS_Line_New_High": rsl_new_high,
        "RS_Line_Near_High": rsl_near_high,
        "RS_Line_Uptrend": rsl_uptrend,
        "RS_Blue_Dot": blue_dot,
    })
    out["TT_Score"] = conds.sum(axis=1).astype(int)
    out["TT_Pass"] = conds.all(axis=1)
    return pd.concat([out, conds], axis=1)


# ---------------- 4b. Stage, base, VCP and volume-breakout detection ----------------
STAGE_NAMES = {0: "n/a", 1: "Stage 1 · Basing", 2: "Stage 2 · Advancing",
               3: "Stage 3 · Topping", 4: "Stage 4 · Declining"}


def classify_stage(p):
    """Weinstein stage from price vs the 150-day SMA (~30-week MA) and its slope.
    2: price above a rising MA            4: price below a falling MA (or >20% below it)
    1: otherwise, MA was falling before   3: otherwise, MA was rising before
    (if the MA was flat before too, the position in the 52-week range decides)."""
    s150 = p.rolling(150).mean()
    k = STAGE_SLOPE_DAYS
    last = p.iloc[-1]
    m0, mk, mp = s150.iloc[-1], s150.iloc[-1 - k], s150.iloc[-1 - k - STAGE_PRIOR_DAYS]
    slope, prior = m0 / mk - 1, mk / mp - 1
    dist = last / m0 - 1
    flat = STAGE_FLAT_SLOPE
    win = p.iloc[-252:]
    pos = (last - win.min()) / (win.max() - win.min())

    s2 = (last > m0) & (slope > flat)
    s4 = (last < m0) & ((slope < -flat) | (dist < -STAGE4_DIST))
    amb = ~s2 & ~s4
    s1 = amb & ((prior < -flat) | ((prior.abs() <= flat) & (pos < 0.5)))
    s3 = amb & ~s1
    num = pd.Series(np.select([s2, s4, s1, s3], [2, 4, 1, 3], default=0), index=p.columns)
    num[m0.isna() | mk.isna() | mp.isna()] = 0
    return pd.DataFrame({"Stage_Num": num,
                         "Stage": num.map(STAGE_NAMES),
                         "SMA150_Slope_%": (slope * 100).round(1)})


def detect_base(close, high, low):
    """Longest recent window (6-30 weeks) that is a sideways consolidation:
    depth <= MAX_BASE_DEPTH and price within +/-MAX_BASE_DRIFT of where the window began."""
    cols = close.columns
    days = pd.Series(0, index=cols)
    depth = pd.Series(np.nan, index=cols)
    pivot = pd.Series(np.nan, index=cols)
    last = close.iloc[-1]
    for w in BASE_WINDOWS:                       # ascending, so the longest valid window wins
        hi, lo = high.iloc[-w:].max(), low.iloc[-w:].min()
        d = (hi - lo) / hi
        drift = last / close.iloc[-w] - 1
        ok = (d <= MAX_BASE_DEPTH) & (drift.abs() <= MAX_BASE_DRIFT)
        days = days.where(~ok, w)
        depth = depth.where(~ok, d * 100)
        pivot = pivot.where(~ok, hi)
    return pd.DataFrame({"Base_Forming": days > 0, "Base_Days": days,
                         "Base_Depth_%": depth.round(1), "Base_Pivot": pivot})


def zigzag(c, pct):
    """Swing pivots of a price array: list of (index, price, +1 high / -1 low).
    A swing is confirmed once price reverses by >= pct from the extreme."""
    piv, direction, hi_i, lo_i = [], 0, 0, 0
    for i in range(1, len(c)):
        if direction == 0:
            if c[i] > c[hi_i]:
                hi_i = i
            if c[i] < c[lo_i]:
                lo_i = i
            if c[i] <= c[hi_i] * (1 - pct):
                piv.append((hi_i, c[hi_i], 1)); direction = -1; lo_i = i
            elif c[i] >= c[lo_i] * (1 + pct):
                piv.append((lo_i, c[lo_i], -1)); direction = 1; hi_i = i
        elif direction == 1:
            if c[i] > c[hi_i]:
                hi_i = i
            elif c[i] <= c[hi_i] * (1 - pct):
                piv.append((hi_i, c[hi_i], 1)); direction = -1; lo_i = i
        else:
            if c[i] < c[lo_i]:
                lo_i = i
            elif c[i] >= c[lo_i] * (1 + pct):
                piv.append((lo_i, c[lo_i], -1)); direction = 1; hi_i = i
    if direction == -1:                          # pullback still in progress: provisional low
        piv.append((lo_i, c[lo_i], -1))
    return piv


def find_vcp(c, h, v):
    """VCP shape on one stock. c/h/v = close/high/volume arrays for the last VCP_LOOKBACK sessions.
    Returns (pullback depths in %, pivot price, volume_contracting) or None."""
    if len(c) < 40 or np.isnan(c).any() or np.isnan(h).any():
        return None
    pk = int(np.argmax(c))                       # left peak of the pattern
    if len(c) - pk < 20:
        return None
    c, h, v = c[pk:], h[pk:], v[pk:]
    piv = zigzag(c, ZIGZAG_PCT)
    legs = [(a[0], b[0], (a[1] - b[1]) / a[1])   # (start idx, end idx, depth)
            for a, b in zip(piv[:-1], piv[1:]) if a[2] == 1 and b[2] == -1]
    if len(legs) < MIN_CONTRACTIONS:
        return None
    run = [legs[-1]]                             # trailing run of successively tighter pullbacks
    for lg in reversed(legs[:-1]):
        if run[0][2] <= lg[2] * TIGHTEN_RATIO:
            run.insert(0, lg)
        else:
            break
    while len(run) > MIN_CONTRACTIONS and run[0][2] > MAX_BASE_DEPTH:
        run.pop(0)
    if len(run) < MIN_CONTRACTIONS or run[0][2] > MAX_BASE_DEPTH or run[-1][2] > VCP_LAST_MAX:
        return None
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vols = [np.nanmean(v[a:b + 1]) for a, b, _ in run]
    vol_ok = bool(vols[-1] < vols[0])
    return [round(d * 100) for *_, d in run], float(h[run[-1][0]]), vol_ok


def pattern_table(prices, ohlcv, idx):
    """Stage, base, VCP and volume-breakout columns for the rated stocks `idx`."""
    p = prices[idx]
    high = ohlcv["High"].reindex(index=p.index, columns=idx)
    low = ohlcv["Low"].reindex(index=p.index, columns=idx)
    vol = ohlcv["Volume"].reindex(index=p.index, columns=idx)
    last = p.iloc[-1]

    stage = classify_stage(p)
    base = detect_base(p, high, low)

    # --- VCP shape, stock by stock ---
    L = VCP_LOOKBACK
    C, H, V = (x.iloc[-L:].to_numpy(float) for x in (p, high, vol))
    vcp_pivot = pd.Series(np.nan, index=idx)
    vcp_text = pd.Series("", index=idx)
    vcp_vol_ok = pd.Series(False, index=idx)
    for j, col in enumerate(idx):
        r = find_vcp(C[:, j], H[:, j], V[:, j])
        if r is not None:
            depths, pv, vok = r
            vcp_pivot[col], vcp_text[col], vcp_vol_ok[col] = pv, "→".join(map(str, depths)), vok

    pivot = vcp_pivot.fillna(base["Base_Pivot"])
    pct_to_pivot = ((pivot / last - 1) * 100).round(1)
    vcp = (vcp_pivot.notna() & (stage["Stage_Num"] == 2)
           & (pct_to_pivot >= -10) & (pct_to_pivot <= 15))

    # --- volume ---
    dry = vol.iloc[-10:].mean() / vol.iloc[-50:].mean() <= VOL_DRYUP_RATIO
    vol_x_today = (vol.iloc[-1] / vol.shift(1).rolling(50).mean().iloc[-1]).round(2)
    vcp_setup = vcp & dry & (pct_to_pivot >= 0) & (pct_to_pivot <= VCP_NEAR_PIVOT)

    # --- volume breakout in the last few sessions ---
    prior_hi = high.rolling(BREAKOUT_LOOKBACK).max().shift(1)
    prior_lo = low.rolling(BREAKOUT_LOOKBACK).min().shift(1)
    prior_depth = (prior_hi - prior_lo) / prior_hi
    vr = vol / vol.shift(1).rolling(50).mean()
    day_range = (high - low).where(lambda x: x > 0)
    close_pos = (p - low) / day_range
    brk = (p > prior_hi) & (vr >= BREAKOUT_VOL_X) & (prior_depth <= MAX_BASE_DEPTH) & (close_pos >= 0.5)
    days_ago = pd.Series(np.nan, index=idx)
    brk_vol = pd.Series(np.nan, index=idx)
    for k in range(BREAKOUT_RECENT_DAYS - 1, -1, -1):    # most recent session wins
        ck = brk.iloc[-1 - k]
        days_ago = days_ago.where(~ck, k)
        brk_vol = brk_vol.where(~ck, vr.iloc[-1 - k])

    return pd.DataFrame({
        "Stage": stage["Stage"], "Stage_Num": stage["Stage_Num"],
        "SMA150_Slope_%": stage["SMA150_Slope_%"],
        "Base_Forming": base["Base_Forming"], "Base_Days": base["Base_Days"],
        "Base_Depth_%": base["Base_Depth_%"],
        "Stage1_Base": (stage["Stage_Num"] == 1) & base["Base_Forming"],
        "VCP": vcp, "VCP_Contractions": vcp_text.where(vcp_pivot.notna(), ""),
        "VCP_Vol_Contracting": vcp & vcp_vol_ok,
        "Vol_Dry_Up": dry, "VCP_Setup": vcp_setup,
        "Pivot": pivot.round(2), "Pct_To_Pivot": pct_to_pivot,
        "Vol_Breakout": days_ago.notna(), "Breakout_Days_Ago": days_ago,
        "Breakout_Vol_x": brk_vol.round(2), "Vol_x_50d": vol_x_today,
    })


def download_shares(tickers, workers=8):
    """Shares outstanding per ticker from Yahoo Finance (NaN where Yahoo has none).
    Share counts change rarely, so cache the result (the app keeps it for 7 days)."""
    def one(t):
        try:
            fi = yf.Ticker(t).fast_info
            sh = None
            try:
                sh = fi["shares"]
            except Exception:
                sh = None
            if not sh:                                   # fall back to market cap / last price
                mc, lp = fi["market_cap"], fi["last_price"]
                sh = mc / lp if mc and lp else None
            return t, (float(sh) if sh else np.nan)
        except Exception:
            return t, np.nan
    tickers = list(tickers)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(one, tickers))
    return pd.Series(dict(res), name="Shares").reindex(tickers)


def market_cap_cr(prices, shares, idx):
    """Market cap in Rs crore = shares outstanding x latest close / 1e7 (NaN if shares unknown)."""
    return (shares.reindex(idx) * prices[idx].iloc[-1] / 1e7).round(0)


def liquidity_table(prices, volume, idx):
    """Average daily traded value (close x volume) over the last LIQUIDITY_DAYS sessions, in Rs crore.
    A stock is 'liquid' when that is at least MIN_TRADED_VALUE_CR."""
    vol = volume.reindex(index=prices.index, columns=idx)
    traded = (prices[idx] * vol).iloc[-LIQUIDITY_DAYS:]
    avg_cr = traded.mean() / 1e7                      # 1 crore = 10,000,000
    return pd.DataFrame({"Avg_Value_Cr": avg_cr.round(1),
                         "Liquid": (avg_cr >= MIN_TRADED_VALUE_CR).fillna(False).astype(bool)})


def build_table(prices, bench, meta, ohlcv=None, shares=None):
    """RS rating + Trend Template (+ stage / base / VCP / breakout columns if `ohlcv` is given)."""
    end_date = prices.index[-1]
    prev_date = prices.index[-1 - CHANGE_DAYS]

    wr_now = weighted_return(prices, end_date)
    rs_now = rs_rating(wr_now)
    rs_prev = rs_rating(weighted_return(prices, prev_date))

    tech = technicals(prices, bench, rs_now)

    out = pd.DataFrame({
        "RS_Rating": rs_now,
        "RS_Change": rs_now - rs_prev.reindex(rs_now.index),
        "Weighted_Return_%": wr_now.reindex(rs_now.index).round(2),
    }).join(tech)
    if ohlcv is not None:
        out = out.join(pattern_table(prices, ohlcv, rs_now.index))
        out = out.join(liquidity_table(prices, ohlcv["Volume"], rs_now.index))
    if shares is not None:
        out["Market_Cap_Cr"] = market_cap_cr(prices, shares, out.index)
    out.index.name = "Ticker"
    keep = [c for c in ["Ticker", "Company Name", "Sector", "Symbol", "Series",
                        "Macro Sector", "Industry", "Basic Industry"] if c in meta.columns]
    out = out.reset_index().merge(meta[keep], on="Ticker", how="left")

    front = ["Symbol", "Company Name", "Sector", "RS_Rating", "RS_Change",
             "Weighted_Return_%", "Stage", "TT_Score", "TT_Pass", "Stage1_Base", "VCP",
             "VCP_Setup", "Vol_Breakout", "Liquid", "Avg_Value_Cr", "Market_Cap_Cr", "RS_Line_New_High", "RS_Line_Near_High",
             "RS_Blue_Dot", "RS_Line_Uptrend", "RSI14", "Close", "Pct_From_52W_High",
             "Pct_Above_52W_Low", "SMA50", "SMA150", "SMA200"]
    front = [c for c in front if c in out.columns]
    rest = [c for c in out.columns if c not in front and c != "Ticker"]
    out = out[front + rest]
    return (out.sort_values(["RS_Rating", "Weighted_Return_%"], ascending=False)
               .reset_index(drop=True), end_date)


def industry_strength(result, min_stocks=3, level="Sector", liquid_only=True):
    """Rank sectors / industries / basic industries by the strength of their stocks.
    Uses liquid stocks only (avg 20-day traded value >= Rs 10 crore) unless liquid_only=False.
    level: "Sector" (NSE's 22 sectors, always available), or "Industry" / "Basic Industry" /
    "Macro Sector" when the NSE classification file is loaded.
    Score (1-99) = percentile rank of 50% median RS Rating + 25% share of stocks with RS >= 80
    + 25% share passing the Trend Template. Groups with fewer than `min_stocks` stocks are skipped."""
    if level not in result.columns:
        return pd.DataFrame()
    r = result
    if liquid_only and "Liquid" in r.columns:
        r = r[r["Liquid"]]
    r = r[r[level].notna() & (r[level].astype(str).str.strip() != "")]
    r = r.assign(_leader=lambda d: d["RS_Rating"] >= 80,
                 _a200=lambda d: d["Close"] > d["SMA200"],
                 _s2=lambda d: d["Stage"].eq(STAGE_NAMES[2]) if "Stage" in d.columns else False)
    g = r.groupby(level)
    leaders = (r.sort_values("RS_Rating", ascending=False)
                .groupby(level)["Symbol"].apply(lambda x: ", ".join(x.head(3))))
    out = pd.DataFrame({
        "Stocks": g.size(),
        "Median_RS": g["RS_Rating"].median(),
        "Avg_RS": g["RS_Rating"].mean(),
        "Avg_RS_Change": g["RS_Change"].mean(),
        "Pct_RS80": g["_leader"].mean() * 100,
        "Pct_TT_Pass": g["TT_Pass"].mean() * 100,
        "Pct_Above_200": g["_a200"].mean() * 100,
        "Pct_Stage2": g["_s2"].mean() * 100,
        "TT_Pass_Count": g["TT_Pass"].sum(),
        "Blue_Dots": g["RS_Blue_Dot"].sum(),
        "Avg_Weighted_Return_%": g["Weighted_Return_%"].mean(),
        "Top_Stocks": leaders,
    })
    out.index.name = "Group"
    out = out[out["Stocks"] >= min_stocks]
    if out.empty:
        return out.reset_index()
    pr = lambda x: x.rank(pct=True)
    blend = 0.5 * pr(out["Median_RS"]) + 0.25 * pr(out["Pct_RS80"]) + 0.25 * pr(out["Pct_TT_Pass"])
    out.insert(0, "Industry_Score", rs_rating(blend))
    out = out.sort_values(["Industry_Score", "Median_RS", "Avg_RS"], ascending=False)
    out.insert(0, "Rank", range(1, len(out) + 1))
    for c in ["Median_RS", "Avg_RS", "Avg_RS_Change", "Avg_Weighted_Return_%"]:
        out[c] = out[c].round(1)
    for c in ["Pct_RS80", "Pct_TT_Pass", "Pct_Above_200", "Pct_Stage2"]:
        out[c] = out[c].round(0)
    out[["TT_Pass_Count", "Blue_Dots"]] = out[["TT_Pass_Count", "Blue_Dots"]].astype(int)
    return out.reset_index()


def market_breadth(result, liquid_only=True):
    """Market-breadth snapshot (liquid stocks only by default): advancers/decliners, share of stocks
    above their 50/150/200-day averages, Stage 2, Trend Template and RS >= 80, 52-week highs/lows."""
    r = result
    if liquid_only and "Liquid" in r.columns:
        r = r[r["Liquid"]]
    n = len(r)

    def pct(mask):
        return round(float(mask.mean() * 100), 1) if n else 0.0

    chg = r["Chg_1D_%"]
    return {
        "stocks": n,
        "advancers": int((chg > 0).sum()), "decliners": int((chg < 0).sum()),
        "unchanged": int((chg == 0).sum()),
        "pct_above_50": pct(r["Close"] > r["SMA50"]),
        "pct_above_150": pct(r["Close"] > r["SMA150"]),
        "pct_above_200": pct(r["Close"] > r["SMA200"]),
        "pct_stage2": pct(r["Stage"].eq(STAGE_NAMES[2])) if "Stage" in r.columns else 0.0,
        "pct_tt": pct(r["TT_Pass"]),
        "pct_rs80": pct(r["RS_Rating"] >= 80),
        "pct_rsline_high": pct(r["RS_Line_New_High"]),
        "new_52w_highs": int((r["Pct_From_52W_High"] >= 0).sum()),
        "new_52w_lows": int((r["Pct_Above_52W_Low"] <= 0).sum()),
    }


# ---------------- 5. Web dashboard (single self-contained HTML file) ----------------
DASHBOARD_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Nifty Total Market RS Scanner</title>
<style>
:root{--bg:#fafaf8;--surface:#fff;--text:#1d1d1b;--muted:#6b6b66;--border:#e3e2dc;--hover:#f3f2ee;--accent:#2a5bd7;
--g1:#0f7b4a;--g1b:#d6f0e2;--g2:#2e7d32;--g2b:#e6f4d9;--g3:#8a6d00;--g3b:#fbf0cc;--g4:#6b6b66;--g4b:#ecebe6;--g5:#b3261e;--g5b:#fadedb;}
@media (prefers-color-scheme: dark){:root{--bg:#141413;--surface:#1c1c1a;--text:#ecebe6;--muted:#9a998f;--border:#33332f;--hover:#252523;--accent:#7aa2ff;
--g1:#5fd39a;--g1b:#173a2b;--g2:#9ad67a;--g2b:#243818;--g3:#e5c24d;--g3b:#3a3213;--g4:#9a998f;--g4b:#2a2a28;--g5:#ff8a80;--g5b:#43201d;}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:14px/1.4 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
header{padding:16px 20px 8px}
h1{margin:0;font-size:20px;font-weight:600}
.sub{color:var(--muted);font-size:13px;margin-top:2px}
.tabs{display:flex;gap:6px;flex-wrap:wrap;padding:8px 20px}
.tab{border:1px solid var(--border);background:var(--surface);color:var(--text);padding:6px 12px;border-radius:999px;cursor:pointer;font:inherit}
.tab.on{background:var(--accent);border-color:var(--accent);color:#fff}
.tab .n{opacity:.7;margin-left:6px}
.controls{display:flex;gap:10px;flex-wrap:wrap;align-items:center;padding:4px 20px 10px}
.controls input,.controls select,.controls button{border:1px solid var(--border);background:var(--surface);color:var(--text);padding:6px 10px;border-radius:8px;font:inherit}
.controls input[type=number]{width:70px}
.controls label{color:var(--muted);font-size:13px}
.wrap{margin:0 20px 20px;border:1px solid var(--border);border-radius:10px;overflow:auto;max-height:calc(100vh - 190px);background:var(--surface)}
table{border-collapse:collapse;width:100%;min-width:1100px}
th,td{padding:7px 10px;border-bottom:1px solid var(--border);white-space:nowrap}
th{position:sticky;top:0;background:var(--surface);text-align:right;font-weight:600;font-size:12px;cursor:pointer;user-select:none;z-index:1;border-bottom:1px solid var(--border)}
th.l{text-align:left}
th .ar{opacity:.5;margin-left:4px}
tbody tr:hover{background:var(--hover)}
td.num{text-align:right;font-variant-numeric:tabular-nums}
td.ctr{text-align:center}
td.sym a{color:var(--accent);text-decoration:none;font-weight:600}
td.sym a:hover{text-decoration:underline}
td.txt{color:var(--muted);max-width:240px;overflow:hidden;text-overflow:ellipsis}
.badge{display:inline-block;min-width:34px;text-align:center;padding:1px 8px;border-radius:6px;font-weight:600}
.g1{color:var(--g1);background:var(--g1b)}.g2{color:var(--g2);background:var(--g2b)}.g3{color:var(--g3);background:var(--g3b)}
.g4{color:var(--g4);background:var(--g4b)}.g5{color:var(--g5);background:var(--g5b)}
.up{color:var(--g1)}.dn{color:var(--g5)}
.pass{color:var(--g1);font-weight:600}
.dot{display:inline-block;width:10px;height:10px;border-radius:50%;background:var(--accent)}
.empty{padding:30px;text-align:center;color:var(--muted)}
.foot{padding:0 20px 20px;color:var(--muted);font-size:12px}
</style></head><body>
<header><h1>Nifty Total Market RS Scanner</h1>
<div class="sub" id="sub"></div></header>
<div class="tabs" id="tabs"></div>
<div class="controls">
<input id="q" type="search" placeholder="Search symbol / company">
<label>Min RS <input id="minrs" type="number" min="0" max="99" value="0"></label>
<select id="ind"><option value="">All sectors</option></select>
<button id="csv">Download CSV</button>
<span class="sub" id="count"></span>
</div>
<div class="wrap"><table><thead id="thead"></thead><tbody id="tbody"></tbody></table></div>
<div class="foot">RS Rating = percentile rank (1-99) of the weighted 3/6/9/12-month return (40/20/20/20) within the NSE 500.
TT = Minervini Trend Template conditions met (of 8). RS Line = stock / Nifty 500. Blue dot = RS Line at 52-week high while price is still below its own high.
52-week high/low use closing prices. Symbols link to TradingView. Not investment advice.</div>
<script>
const P = __PAYLOAD__;
const rows = P.rows;
const COLS = [
 {k:'Symbol',t:'Symbol',type:'sym',l:1},
 {k:'Company Name',t:'Company',type:'txt',l:1},
 {k:'Sector',t:'Sector',type:'txt',l:1},
 {k:'RS_Rating',t:'RS',type:'rs'},
 {k:'RS_Change',t:'Chg',type:'chg'},
 {k:'TT_Score',t:'TT',type:'tt'},
 {k:'RS_Line_New_High',t:'RS Line high',type:'flag'},
 {k:'RS_Blue_Dot',t:'Blue dot',type:'dot'},
 {k:'RSI14',t:'RSI',type:'num1'},
 {k:'Close',t:'Close',type:'num2'},
 {k:'Pct_From_52W_High',t:'% from 52W high',type:'pct'},
 {k:'Pct_Above_52W_Low',t:'% above 52W low',type:'pct'},
 {k:'Weighted_Return_%',t:'Wtd return %',type:'pct'}
];
const TABS = {
 top:{label:'Top picks', f:r=>r.TT_Pass && r.RS_Rating>=P.pref && r.RS_Line_Near_High && r.RSI14>50},
 tt:{label:'Trend Template pass', f:r=>r.TT_Pass},
 blue:{label:'RS blue dot', f:r=>r.RS_Blue_Dot && r.RS_Rating>=P.pref},
 all:{label:'All stocks', f:r=>true}
};
let tab='top', sortK='RS_Rating', sortD=-1;
const $ = id => document.getElementById(id);
const esc = s => String(s==null?'':s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const fmt = (v,d) => (v==null?'-':Number(v).toFixed(d));

$('sub').textContent = 'Data as of ' + P.asof + ' | ' + rows.length + ' stocks rated | Top picks = Trend Template + RS >= ' + P.pref + ' + RS Line near high + RSI > 50';
[...new Set(rows.map(r=>r.Sector).filter(Boolean))].sort().forEach(i=>{
  const o=document.createElement('option'); o.value=i; o.textContent=i; $('ind').appendChild(o);
});

function drawTabs(){
  $('tabs').innerHTML = Object.entries(TABS).map(([k,t])=>
    `<button class="tab ${k===tab?'on':''}" data-k="${k}">${t.label}<span class="n">${rows.filter(t.f).length}</span></button>`).join('');
  document.querySelectorAll('.tab').forEach(b=>b.onclick=()=>{tab=b.dataset.k;drawTabs();draw();});
}
function drawHead(){
  $('thead').innerHTML = '<tr>' + COLS.map(c=>
    `<th class="${c.l?'l':''}" data-k="${c.k}">${c.t}<span class="ar">${c.k===sortK?(sortD>0?'\u25B2':'\u25BC'):''}</span></th>`).join('') + '</tr>';
  document.querySelectorAll('th').forEach(th=>th.onclick=()=>{
    const k=th.dataset.k; if(k===sortK) sortD=-sortD; else {sortK=k; sortD=th.classList.contains('l')?1:-1;}
    drawHead(); draw();
  });
}
function cell(c,r){
  const v=r[c.k];
  switch(c.type){
    case 'sym': return `<td class="sym"><a href="https://www.tradingview.com/chart/?symbol=NSE%3A${encodeURIComponent(v)}" target="_blank" rel="noopener">${esc(v)}</a></td>`;
    case 'txt': return `<td class="txt" title="${esc(v)}">${esc(v)}</td>`;
    case 'rs': {const cls=v>=90?'g1':v>=80?'g2':v>=70?'g3':v>=50?'g4':'g5'; return `<td class="num"><span class="badge ${cls}">${v}</span></td>`;}
    case 'chg': return `<td class="num ${v>0?'up':v<0?'dn':''}">${v>0?'+':''}${v==null?'-':v}</td>`;
    case 'tt': return `<td class="num"><span class="${r.TT_Pass?'pass':''}">${v}/8</span></td>`;
    case 'flag': return `<td class="ctr">${v?'\u2713':''}</td>`;
    case 'dot': return `<td class="ctr">${v?'<span class="dot" title="RS Line at 52W high before price"></span>':''}</td>`;
    case 'num1': return `<td class="num">${fmt(v,1)}</td>`;
    case 'num2': return `<td class="num">${fmt(v,2)}</td>`;
    default: return `<td class="num">${fmt(v,1)}</td>`;
  }
}
function filtered(){
  const q=$('q').value.trim().toLowerCase(), minrs=+$('minrs').value||0, ind=$('ind').value;
  let out = rows.filter(TABS[tab].f).filter(r=>r.RS_Rating>=minrs && (!ind||r.Sector===ind) &&
    (!q || (r.Symbol||'').toLowerCase().includes(q) || (r['Company Name']||'').toLowerCase().includes(q)));
  out.sort((a,b)=>{
    let x=a[sortK], y=b[sortK];
    if(x==null&&y==null) return 0; if(x==null) return 1; if(y==null) return -1;
    if(typeof x==='string') return sortD*x.localeCompare(y);
    return sortD*(Number(x)-Number(y));
  });
  return out;
}
function draw(){
  const f=filtered();
  $('count').textContent = f.length + ' shown';
  $('tbody').innerHTML = f.length ? f.map(r=>'<tr>'+COLS.map(c=>cell(c,r)).join('')+'</tr>').join('')
    : `<tr><td class="empty" colspan="${COLS.length}">No stocks match these filters.</td></tr>`;
}
$('csv').onclick = ()=>{
  const f=filtered(), keys=Object.keys(rows[0]||{});
  const csv=[keys.join(',')].concat(f.map(r=>keys.map(k=>{const v=r[k]; return v==null?'':'"'+String(v).replace(/"/g,'""')+'"';}).join(','))).join('\n');
  const a=document.createElement('a'); a.href=URL.createObjectURL(new Blob([csv],{type:'text/csv'}));
  a.download='nse500_'+tab+'_'+P.asof+'.csv'; a.click();
};
['q','minrs','ind'].forEach(id=>$(id).addEventListener('input',draw));
drawTabs(); drawHead(); draw();
</script></body></html>
"""


def build_dashboard(result, as_of, path="nse500_rs_dashboard.html"):
    """Write a self-contained, offline HTML page (sortable/filterable tables)."""
    payload = {
        "asof": str(as_of.date()),
        "pref": PREFERRED_RS,
        "rows": json.loads(result.to_json(orient="records")),   # NaN -> null
    }
    blob = json.dumps(payload, allow_nan=False).replace("</", "<\\/")
    with open(path, "w", encoding="utf-8") as f:
        f.write(DASHBOARD_TEMPLATE.replace("__PAYLOAD__", blob))
    return path


# ---------------- 6. Run ----------------
def main():
    meta = load_universe()
    if meta is None:                       # last resort in Colab: upload the NSE list by hand
        print(f"Please upload '{LOCAL_UNIVERSE_FILE}' (Nifty Total Market list from nseindia.com).")
        from google.colab import files
        up = files.upload()
        meta = parse_universe(pd.read_csv(io.BytesIO(next(iter(up.values())))))
    meta = attach_classification(meta, load_classification())
    n_cls = int(meta["Basic Industry"].notna().sum())
    print(f"Universe: {len(meta)} stocks ({UNIVERSE_NAME}); NSE industry classification for {n_cls} of them")
    if n_cls == 0:
        print("  (tip: run build_classification_file() once to add NSE industry / basic industry)")

    md = download_market_data(meta["Ticker"].tolist())
    prices = md["Close"]
    bench = get_benchmark()
    print(f"Price data for {prices.shape[1]} stocks, last date: {prices.index[-1].date()}")
    print("Fetching shares outstanding for market cap (about a minute)...")
    shares = download_shares(prices.columns)

    result, as_of = build_table(prices, bench, meta, ohlcv=md, shares=shares)
    assert len(result) > 0, "No ratings produced - check price data above"
    n_liq = int(result["Liquid"].sum())
    print(f"\nRS Ratings as of {as_of.date()} (scan size: {len(result)} stocks)")
    print(f"Liquid stocks (avg {LIQUIDITY_DAYS}-day traded value >= Rs {MIN_TRADED_VALUE_CR:.0f} crore): "
          f"{n_liq} of {len(result)}")
    skipped = prices.shape[1] - len(result)
    if skipped:
        print(f"({skipped} stocks skipped: less than 12 months of history)")

    show = ["Symbol", "RS_Rating", "RS_Change", "TT_Score", "Liquid", "Avg_Value_Cr", "RS_Line_New_High",
            "RS_Blue_Dot", "RSI14", "Close", "Pct_From_52W_High", "Pct_Above_52W_Low"]
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", None)

    b = market_breadth(result)
    print(f"\n=== MARKET BREADTH (liquid stocks only: {b['stocks']}) ===")
    print(f"Advancers {b['advancers']} | Decliners {b['decliners']} | Unchanged {b['unchanged']}")
    print(f"Above 50 SMA {b['pct_above_50']}% | above 150 SMA {b['pct_above_150']}% | above 200 SMA {b['pct_above_200']}%")
    print(f"Stage 2 {b['pct_stage2']}% | Trend Template {b['pct_tt']}% | RS>=80 {b['pct_rs80']}%")
    print(f"New 52-week highs {b['new_52w_highs']} | new 52-week lows {b['new_52w_lows']}")

    # stock lists use the market-cap filter (breadth / sector figures above do not)
    scan = result
    n_cap = int(result["Market_Cap_Cr"].notna().sum())
    if MIN_MARKET_CAP_CR > 0 and n_cap >= 0.5 * len(result):
        scan = result[result["Market_Cap_Cr"] > MIN_MARKET_CAP_CR]
        print(f"\nMarket-cap filter: {len(scan)} of {len(result)} stocks above Rs {MIN_MARKET_CAP_CR:.0f} crore "
              f"({len(result) - n_cap} had no market cap and are excluded)")
    elif MIN_MARKET_CAP_CR > 0:
        print(f"\nMarket cap known for only {n_cap} of {len(result)} stocks - filter skipped.")
    tt = scan[scan.TT_Pass]
    top = scan[scan.TT_Pass & (scan.RS_Rating >= PREFERRED_RS)
               & scan.RS_Line_Near_High & (scan.RSI14 > 50)]
    blue = scan[scan.RS_Blue_Dot & (scan.RS_Rating >= PREFERRED_RS)]

    print(f"\n=== Passing all 8 Minervini conditions: {len(tt)} stocks ===")
    print(tt[show].head(40).to_string(index=False))
    print(f"\n=== TOP PICKS: Trend Template + RS>={PREFERRED_RS} + RS Line near high + RSI>50: {len(top)} ===")
    print(top[show].head(40).to_string(index=False))
    print(f"\n=== RS BLUE DOT (RS line at 52w high, price not yet): {len(blue)} ===")
    print(blue[show].head(40).to_string(index=False))

    pat_cols = ["Symbol", "RS_Rating", "Stage", "VCP_Contractions", "Vol_Dry_Up", "Pivot",
                "Pct_To_Pivot", "Breakout_Days_Ago", "Breakout_Vol_x", "Base_Days", "Liquid", "Market_Cap_Cr", "Close"]
    setups = scan[scan.VCP_Setup]
    brk = scan[scan.Vol_Breakout]
    s1 = scan[scan.Stage1_Base].sort_values("Base_Days", ascending=False)
    print(f"\n=== VCP SETUPS (tight, volume dried up, within {VCP_NEAR_PIVOT:.0f}% of pivot): {len(setups)} ===")
    print(setups[pat_cols].head(30).to_string(index=False))
    print(f"\n=== VOLUME BREAKOUTS (last {BREAKOUT_RECENT_DAYS} sessions): {len(brk)} ===")
    print(brk[pat_cols].head(30).to_string(index=False))
    print(f"\n=== STAGE 1 BASES FORMING: {len(s1)} ===")
    print(s1[pat_cols].head(30).to_string(index=False))
    scan[scan.VCP | scan.Vol_Breakout | scan.Stage1_Base].to_csv(
        "nse500_vcp_breakout_stage1.csv", index=False)

    for level in ("Sector", "Industry", "Basic Industry"):
        ind = industry_strength(result, level=level)
        if ind.empty:
            continue
        print(f"\n=== {level.upper()} STRENGTH - liquid stocks only (top 15 of {len(ind)}) ===")
        print(ind[["Rank", "Group", "Industry_Score", "Stocks", "Median_RS", "Pct_RS80",
                   "Pct_TT_Pass", "Top_Stocks"]].head(15).to_string(index=False))
        ind.to_csv(f"nse500_{level.lower().replace(' ', '_')}_strength.csv", index=False)

    result.to_csv("nse500_rs_rating_full.csv", index=False)
    tt.to_csv("nse500_trend_template_pass.csv", index=False)
    top.to_csv("nse500_top_picks.csv", index=False)
    print("\nSaved: nse500_rs_rating_full.csv, nse500_trend_template_pass.csv, nse500_top_picks.csv")

    html_path = build_dashboard(result, as_of)
    print(f"Web dashboard saved: {html_path}  (open it in any browser)")
    try:                                   # auto-download when running in Colab
        from google.colab import files
        files.download(html_path)
    except Exception:
        pass
    return result


if __name__ == "__main__":
    result = main()

# To download files in Colab:
# from google.colab import files
# for f in ["nse500_rs_rating_full.csv", "nse500_trend_template_pass.csv", "nse500_top_picks.csv"]:
#     files.download(f)
#
# Look up one stock:
# result[result.Symbol == "RELIANCE"].T
