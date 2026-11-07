# ============================================================
# NSE 500 RS Rating + Minervini Trend Template scanner
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
# Extras: RS Line (stock / Nifty 500) new-high flags, RS "blue dot"
#         (RS line at 52w high before price), RSI(14)
# ============================================================

# --- Cell 1: install (run once in Colab) ---
# !pip install -q yfinance

import io
import json
import time
import requests
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

NSE500_URL = "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv"


# ---------------- 1. NSE 500 constituents ----------------
def get_nse500_symbols():
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    try:
        r = requests.get(NSE500_URL, headers=headers, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
    except Exception as e:
        print(f"Could not download list from NSE ({e}).")
        print("Please upload 'ind_nifty500list.csv' (from niftyindices.com).")
        from google.colab import files
        up = files.upload()
        df = pd.read_csv(io.BytesIO(next(iter(up.values()))))
    df.columns = [c.strip() for c in df.columns]
    return df[["Company Name", "Industry", "Symbol"]].assign(
        Ticker=lambda d: d["Symbol"].str.strip() + ".NS"
    )


# ---------------- 2. Price download ----------------
def _clean_index(obj):
    obj.index = pd.to_datetime(obj.index)
    if obj.index.tz is not None:
        obj.index = obj.index.tz_localize(None)
    obj.index = obj.index.normalize()
    return obj


def download_prices(tickers):
    frames = []
    for i in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[i:i + BATCH_SIZE]
        print(f"Downloading {i + 1}-{i + len(batch)} of {len(tickers)} ...")
        data = yf.download(batch, period=HISTORY_PERIOD, interval="1d",
                           auto_adjust=True, progress=False, threads=True)
        close = data["Close"]
        if isinstance(close, pd.Series):
            close = close.to_frame(batch[0])
        frames.append(close)
        time.sleep(1)
    prices = pd.concat(frames, axis=1)
    prices = prices.loc[:, ~prices.columns.duplicated()]
    prices = _clean_index(prices.sort_index().dropna(axis=1, how="all"))

    # drop stale tickers (delisted / merged / suspended)
    last_valid = prices.apply(lambda s: s.last_valid_index())
    stale = last_valid < prices.index[-1] - pd.Timedelta(days=10)
    if stale.any():
        print(f"Dropping {stale.sum()} stale tickers: {list(prices.columns[stale])[:10]}")
    return prices.loc[:, ~stale].ffill()


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
        "RS_Line_New_High": rsl_new_high,
        "RS_Line_Near_High": rsl_near_high,
        "RS_Line_Uptrend": rsl_uptrend,
        "RS_Blue_Dot": blue_dot,
    })
    out["TT_Score"] = conds.sum(axis=1).astype(int)
    out["TT_Pass"] = conds.all(axis=1)
    return pd.concat([out, conds], axis=1)


def build_table(prices, bench, meta):
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
    out.index.name = "Ticker"
    out = out.reset_index().merge(meta, on="Ticker", how="left")

    front = ["Symbol", "Company Name", "Industry", "RS_Rating", "RS_Change",
             "Weighted_Return_%", "TT_Score", "TT_Pass", "RS_Line_New_High",
             "RS_Line_Near_High", "RS_Blue_Dot", "RS_Line_Uptrend", "RSI14",
             "Close", "Pct_From_52W_High", "Pct_Above_52W_Low",
             "SMA50", "SMA150", "SMA200"]
    rest = [c for c in out.columns if c not in front and c != "Ticker"]
    out = out[front + rest]
    return (out.sort_values(["RS_Rating", "Weighted_Return_%"], ascending=False)
               .reset_index(drop=True), end_date)


def industry_strength(result, min_stocks=3):
    """Rank industries by how strong their stocks are.
    Industry score (1-99) = percentile rank of a blend of:
      50% median RS Rating, 25% share of stocks with RS >= 80,
      25% share of stocks passing the Trend Template.
    Industries with fewer than `min_stocks` rated stocks are skipped (too noisy)."""
    r = result.dropna(subset=["Industry"]).assign(_leader=lambda d: d["RS_Rating"] >= 80)
    g = r.groupby("Industry")
    leaders = (r.sort_values("RS_Rating", ascending=False)
                .groupby("Industry")["Symbol"].apply(lambda s: ", ".join(s.head(3))))
    out = pd.DataFrame({
        "Stocks": g.size(),
        "Median_RS": g["RS_Rating"].median(),
        "Avg_RS": g["RS_Rating"].mean(),
        "Avg_RS_Change": g["RS_Change"].mean(),
        "Pct_RS80": g["_leader"].mean() * 100,
        "Pct_TT_Pass": g["TT_Pass"].mean() * 100,
        "TT_Pass_Count": g["TT_Pass"].sum(),
        "Blue_Dots": g["RS_Blue_Dot"].sum(),
        "Avg_Weighted_Return_%": g["Weighted_Return_%"].mean(),
        "Top_Stocks": leaders,
    })
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
    for c in ["Pct_RS80", "Pct_TT_Pass"]:
        out[c] = out[c].round(0)
    out[["TT_Pass_Count", "Blue_Dots"]] = out[["TT_Pass_Count", "Blue_Dots"]].astype(int)
    return out.reset_index()


# ---------------- 5. Web dashboard (single self-contained HTML file) ----------------
DASHBOARD_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NSE 500 RS Rating Scanner</title>
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
<header><h1>NSE 500 RS Rating Scanner</h1>
<div class="sub" id="sub"></div></header>
<div class="tabs" id="tabs"></div>
<div class="controls">
<input id="q" type="search" placeholder="Search symbol / company">
<label>Min RS <input id="minrs" type="number" min="0" max="99" value="0"></label>
<select id="ind"><option value="">All industries</option></select>
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
 {k:'Industry',t:'Industry',type:'txt',l:1},
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
[...new Set(rows.map(r=>r.Industry).filter(Boolean))].sort().forEach(i=>{
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
  let out = rows.filter(TABS[tab].f).filter(r=>r.RS_Rating>=minrs && (!ind||r.Industry===ind) &&
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
    meta = get_nse500_symbols()
    print(f"Universe: {len(meta)} stocks")

    prices = download_prices(meta["Ticker"].tolist())
    bench = get_benchmark()
    print(f"Price data for {prices.shape[1]} stocks, last date: {prices.index[-1].date()}")

    result, as_of = build_table(prices, bench, meta)
    assert len(result) > 0, "No ratings produced - check price data above"
    print(f"\nRS Ratings as of {as_of.date()} (scan size: {len(result)} stocks)")
    skipped = prices.shape[1] - len(result)
    if skipped:
        print(f"({skipped} stocks skipped: less than 12 months of history)")

    show = ["Symbol", "RS_Rating", "RS_Change", "TT_Score", "RS_Line_New_High",
            "RS_Blue_Dot", "RSI14", "Close", "Pct_From_52W_High", "Pct_Above_52W_Low"]
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", None)

    tt = result[result.TT_Pass]
    top = result[result.TT_Pass & (result.RS_Rating >= PREFERRED_RS)
                 & result.RS_Line_Near_High & (result.RSI14 > 50)]
    blue = result[result.RS_Blue_Dot & (result.RS_Rating >= PREFERRED_RS)]

    print(f"\n=== Passing all 8 Minervini conditions: {len(tt)} stocks ===")
    print(tt[show].head(40).to_string(index=False))
    print(f"\n=== TOP PICKS: Trend Template + RS>={PREFERRED_RS} + RS Line near high + RSI>50: {len(top)} ===")
    print(top[show].head(40).to_string(index=False))
    print(f"\n=== RS BLUE DOT (RS line at 52w high, price not yet): {len(blue)} ===")
    print(blue[show].head(40).to_string(index=False))

    ind = industry_strength(result)
    print(f"\n=== INDUSTRY STRENGTH (top 15 of {len(ind)} industries) ===")
    print(ind[["Rank", "Industry", "Industry_Score", "Stocks", "Median_RS", "Pct_RS80",
               "Pct_TT_Pass", "Top_Stocks"]].head(15).to_string(index=False))
    ind.to_csv("nse500_industry_strength.csv", index=False)

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
