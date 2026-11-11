"""Look & feel for the Nifty Total Market scanner: fonts, global CSS, hero, KPI cards and
sortable, responsive HTML tables that follow Streamlit's light/dark theme."""
import json

import streamlit as st
import streamlit.components.v1 as components

# ---------------------------------------------------------------- fonts
FONTS = {
    "Inter": "Inter",
    "Poppins": "Poppins",
    "DM Sans": "DM Sans",
    "Manrope": "Manrope",
    "IBM Plex Sans": "IBM Plex Sans",
    "Space Grotesk": "Space Grotesk",
    "Source Sans 3": "Source Sans 3",
    "System default": None,
}
_FALLBACK = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"


def font_url(name: str) -> str:
    fam = FONTS.get(name)
    if not fam:
        return ""
    return (f"https://fonts.googleapis.com/css2?family={fam.replace(' ', '+')}"
            f":wght@400;500;600;700&display=swap")


def font_family(name: str) -> str:
    fam = FONTS.get(name)
    return f"'{fam}', {_FALLBACK}" if fam else _FALLBACK


# ---------------------------------------------------------------- page CSS
_PAGE_CSS = """
<style>
__IMPORT__
:root{--ui-font:__FAMILY__;--accent:#6366f1;--accent2:#ec4899;
--line:color-mix(in srgb,currentColor 13%,transparent);--soft:color-mix(in srgb,currentColor 5%,transparent)}
.stApp,.stApp div,.stApp p,.stApp label,.stApp li,.stApp h1,.stApp h2,.stApp h3,.stApp h4,
.stApp input,.stApp textarea,.stApp button,.stApp a,.stApp td,.stApp th{font-family:var(--ui-font)!important}
.block-container{padding-top:1.4rem!important;padding-bottom:3rem!important;max-width:1600px}
header[data-testid="stHeader"]{background:transparent}
footer{visibility:hidden}
[data-testid="stAppDeployButton"],.stAppDeployButton{display:none!important}
div[data-testid="stElementContainer"]:has(> div > style),.element-container:has(> style){display:none}

/* hero */
.hero{display:flex;justify-content:space-between;align-items:flex-end;gap:16px;flex-wrap:wrap;margin:2px 0 18px}
.eyebrow{font-size:.72rem;letter-spacing:.14em;text-transform:uppercase;opacity:.6;font-weight:600}
.title{font-size:clamp(1.7rem,3.4vw,2.6rem);font-weight:700;line-height:1.1;margin:.25rem 0 .4rem;
background:linear-gradient(90deg,var(--accent),var(--accent2));-webkit-background-clip:text;
background-clip:text;color:transparent!important;-webkit-text-fill-color:transparent;letter-spacing:-.02em}
.sub{opacity:.72;font-size:.95rem;max-width:780px;line-height:1.5}
.chips{display:flex;gap:8px;flex-wrap:wrap}
.chip{border:1px solid var(--line);background:var(--soft);border-radius:999px;padding:5px 12px;
font-size:.78rem;white-space:nowrap}
.chip b{font-weight:600}

/* KPI cards */
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(128px,1fr));gap:12px;margin:4px 0 20px}
.kpi{position:relative;overflow:hidden;border-radius:14px;padding:14px 16px 13px 20px;
border:1px solid var(--line);background:var(--soft)}
.kpi:before{content:"";position:absolute;left:0;top:0;bottom:0;width:5px;background:var(--kc,var(--accent))}
.kpi .l{font-size:.7rem;letter-spacing:.08em;text-transform:uppercase;opacity:.62;font-weight:600}
.kpi .v{font-size:2rem;font-weight:700;line-height:1.15;letter-spacing:-.02em;margin-top:2px}
.kpi .s{font-size:.75rem;opacity:.55;margin-top:1px}

/* tabs as pills (works with old and new Streamlit tab markup) */
[role="tablist"]{gap:8px!important;overflow-x:auto;border-bottom:none!important;box-shadow:none!important;
padding:2px 2px 10px;scrollbar-width:thin}
@media (min-width:900px){[role="tablist"]{flex-wrap:wrap!important;overflow:visible!important}}
[role="tablist"] [role="tab"]{border-radius:999px!important;padding:7px 16px!important;height:auto!important;
white-space:nowrap;background:var(--soft)!important;border:1px solid var(--line)!important;cursor:pointer}
[role="tablist"] [role="tab"] p{font-size:.88rem;font-weight:600;margin:0}
[role="tablist"] [role="tab"][aria-selected="true"]{background:var(--accent)!important;border-color:var(--accent)!important}
[role="tablist"] [role="tab"][aria-selected="true"] p{color:#fff!important}
.react-aria-SelectionIndicator,[data-baseweb="tab-highlight"],[data-baseweb="tab-border"]{display:none!important}

/* sidebar + widgets */
section[data-testid="stSidebar"]{border-right:1px solid var(--line)}
section[data-testid="stSidebar"] h2,section[data-testid="stSidebar"] h3{font-size:1rem;letter-spacing:-.01em}
.stButton>button,.stDownloadButton>button{border-radius:10px;border:1px solid var(--line);font-weight:600}
.stDownloadButton>button{background:var(--soft)}
div[data-baseweb="select"]>div,.stTextInput input{border-radius:10px!important}
.tabnote{opacity:.72;font-size:.88rem;line-height:1.5;margin:2px 0 12px;max-width:1000px}
.tabnote b{font-weight:600}
@media (max-width:640px){.block-container{padding-top:3.4rem!important;padding-left:.8rem!important;padding-right:.8rem!important}
.kpi .v{font-size:1.6rem}.chip{font-size:.72rem}}
</style>
"""


def inject_css(font: str) -> None:
    css = (_PAGE_CSS
           .replace("__IMPORT__", f"@import url('{font_url(font)}');" if font_url(font) else "")
           .replace("__FAMILY__", font_family(font)))
    st.markdown(css, unsafe_allow_html=True)


def hero(title: str, subtitle: str, chips: list) -> None:
    chip_html = "".join(f'<span class="chip">{c}</span>' for c in chips)
    st.markdown(
        f'<div class="hero"><div><div class="eyebrow">Nifty Total Market · Relative strength</div>'
        f'<div class="title">{title}</div><div class="sub">{subtitle}</div></div>'
        f'<div class="chips">{chip_html}</div></div>', unsafe_allow_html=True)


def kpis(items: list) -> None:
    """items: (label, value, sub, colour)"""
    cards = "".join(
        f'<div class="kpi" style="--kc:{c}"><div class="l">{l}</div><div class="v">{v}</div>'
        f'<div class="s">{s}</div></div>' for l, v, s, c in items)
    st.markdown(f'<div class="kpis">{cards}</div>', unsafe_allow_html=True)


def note(text: str) -> None:
    st.markdown(f'<div class="tabnote">{text}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------- table columns
# type: sym txt rs chg sgn1 stage tt chk dot n0 n1 n2 pct spct bar text ago mult
# pri: 1 always shown | 2 hidden on small phones | 3 hidden on phones (tables scroll sideways otherwise)
SPEC = {
    "Symbol": dict(t="Symbol", type="sym", pri=1, left=True),
    "Company Name": dict(t="Company", type="txt", pri=3, left=True),
    "Sector": dict(t="Sector", type="txt", pri=3, left=True),
    "Basic Industry": dict(t="Basic industry", type="txt", pri=3, left=True),
    "Liquid": dict(t="Liquid", type="chk", pri=2),
    "Avg_Traded_Value_20D_Cr": dict(t="20D ADTV ₹Cr", type="n1", pri=2),
    "Market_Cap_Cr": dict(t="Mkt cap ₹Cr", type="n1", pri=2),
    "RS_Rating": dict(t="RS", type="rs", pri=1),
    "RS_Change": dict(t="Chg", type="chg", pri=2),
    "Stage": dict(t="Stage", type="stage", pri=2, left=True),
    "TT_Score": dict(t="Trend Template", type="tt", pri=2),
    "RS_Line_New_High": dict(t="RS line high", type="chk", pri=3),
    "RS_Blue_Dot": dict(t="Blue dot", type="dot", pri=3),
    "RSI14": dict(t="RSI", type="n1", pri=2),
    "Close": dict(t="Close", type="n2", pri=1),
    "Pct_From_52W_High": dict(t="From 52W high", type="pct", pri=2),
    "Pct_Above_52W_Low": dict(t="Above 52W low", type="pct", pri=3),
    "Weighted_Return_%": dict(t="Weighted return", type="spct", pri=3),
    "VCP": dict(t="VCP", type="chk", pri=2),
    "VCP_Contractions": dict(t="Pullbacks %", type="text", pri=2, left=True),
    "VCP_Vol_Contracting": dict(t="Vol shrinking", type="chk", pri=3),
    "Vol_Dry_Up": dict(t="Vol dry-up", type="chk", pri=3),
    "VCP_Setup": dict(t="VCP setup", type="chk", pri=2),
    "Pivot": dict(t="Pivot", type="n2", pri=2),
    "Pct_To_Pivot": dict(t="To pivot", type="pct", pri=1),
    "Vol_Breakout": dict(t="Breakout", type="chk", pri=2),
    "Breakout_Days_Ago": dict(t="When", type="ago", pri=2),
    "Breakout_Vol_x": dict(t="Breakout vol", type="mult", pri=2),
    "Vol_x_50d": dict(t="Vol today", type="mult", pri=3),
    "Stage1_Base": dict(t="Stage 1 base", type="chk", pri=3),
    "Base_Days": dict(t="Base length", type="days", pri=2),
    "Base_Depth_%": dict(t="Base depth", type="pct", pri=3),
    # industry table
    "Rank": dict(t="#", type="n0", pri=1),
    "Industry_Score": dict(t="Strength", type="bar", pri=1),
    "Stocks": dict(t="Stocks", type="n0", pri=2),
    "Median_RS": dict(t="Median RS", type="n0", pri=2),
    "Avg_RS": dict(t="Avg RS", type="n1", pri=3),
    "Avg_RS_Change": dict(t="Avg chg", type="sgn1", pri=3),
    "Pct_RS80": dict(t="RS ≥ 80", type="pct0", pri=2),
    "Pct_TT_Pass": dict(t="Trend Template", type="pct0", pri=2),
    "TT_Pass_Count": dict(t="TT pass", type="n0", pri=3),
    "Blue_Dots": dict(t="Blue dots", type="n0", pri=3),
    "Avg_Weighted_Return_%": dict(t="Avg return", type="spct", pri=3),
    "Top_Stocks": dict(t="Leaders", type="text", pri=2, left=True),
}
COLS_DEFAULT = ["Symbol", "Company Name", "Sector", "Basic Industry", "Liquid",
                "Avg_Traded_Value_20D_Cr", "Market_Cap_Cr", "RS_Rating", "RS_Change", "Stage", "TT_Score",
                "RS_Line_New_High", "RS_Blue_Dot", "RSI14", "Close", "Pct_From_52W_High",
                "Pct_Above_52W_Low", "Weighted_Return_%"]
COLS_PATTERN = ["Symbol", "RS_Rating", "Liquid", "Avg_Traded_Value_20D_Cr", "Market_Cap_Cr", "Stage", "TT_Score", "VCP", "VCP_Contractions", "Vol_Dry_Up",
                "VCP_Setup", "Pivot", "Pct_To_Pivot", "Close", "Vol_Breakout", "Breakout_Days_Ago",
                "Breakout_Vol_x", "Vol_x_50d", "Stage1_Base", "Base_Days", "Base_Depth_%", "Basic Industry",
                "Company Name"]
COLS_INDUSTRY = ["Rank", "Basic Industry", "Industry_Score", "Stocks", "Median_RS", "Avg_RS", "Avg_RS_Change",
                 "Pct_RS80", "Pct_TT_Pass", "TT_Pass_Count", "Blue_Dots", "Avg_Weighted_Return_%",
                 "Top_Stocks"]

# ---------------------------------------------------------------- table iframe
_TABLE_HTML = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
__IMPORT__
:root{--fg:#31333f;--bg:#fff;--accent:#6366f1;--accent2:#ec4899;--rowh:__ROWH__px;--ff:__FAMILY__;
--line:color-mix(in srgb,var(--fg) 14%,transparent);--soft:color-mix(in srgb,var(--fg) 6%,transparent);
--g:#047857;--a:#92400e;--r:#b91c1c;--b:#1d4ed8}
html.dark{--g:#34d399;--a:#fbbf24;--r:#f87171;--b:#60a5fa}
*{box-sizing:border-box}
html,body{margin:0;height:100%;background:var(--bg);color:var(--fg);font:13.5px/1.35 var(--ff)}
.wrap{height:100vh;overflow:auto;border:1px solid var(--line);border-radius:14px}
table{border-collapse:separate;border-spacing:0;width:100%;min-width:max-content}
th{position:sticky;top:0;z-index:3;padding:0 12px;height:42px;text-align:right;white-space:nowrap;cursor:pointer;
user-select:none;font-size:.7rem;text-transform:uppercase;letter-spacing:.06em;font-weight:600;
color:color-mix(in srgb,var(--fg) 62%,var(--bg));background:color-mix(in srgb,var(--fg) 5%,var(--bg));
border-bottom:1px solid var(--line)}
th:hover{color:var(--fg)}
th.l,td.l{text-align:left}
th .ar{display:inline-block;width:12px;color:var(--accent)}
td{height:var(--rowh);padding:0 12px;white-space:nowrap;text-align:right;border-bottom:1px solid var(--line);
font-variant-numeric:tabular-nums}
tr:last-child td{border-bottom:none}
tbody tr:hover td{background:color-mix(in srgb,var(--accent) 8%,var(--bg))}
th:first-child,td:first-child{position:sticky;left:0;z-index:2;background:var(--bg)}
th:first-child{z-index:4;background:color-mix(in srgb,var(--fg) 5%,var(--bg))}
a.sym{color:var(--accent);font-weight:700;text-decoration:none}
a.sym:hover{text-decoration:underline}
.ell{display:inline-block;max-width:210px;overflow:hidden;text-overflow:ellipsis;vertical-align:middle;
color:color-mix(in srgb,var(--fg) 70%,var(--bg))}
.badge{display:inline-block;min-width:38px;text-align:center;padding:2px 9px;border-radius:999px;font-weight:700}
.c1{background:#059669;color:#fff}
.c2{background:color-mix(in srgb,#10b981 22%,var(--bg));color:var(--g)}
.c3{background:color-mix(in srgb,#f59e0b 24%,var(--bg));color:var(--a)}
html.dark .c3{color:#fbbf24}
.c4{background:var(--soft);color:color-mix(in srgb,var(--fg) 75%,var(--bg))}
.c5{background:color-mix(in srgb,#ef4444 18%,var(--bg));color:var(--r)}
.chipx{display:inline-block;padding:2px 10px;border-radius:8px;font-size:.78rem;font-weight:600}
.s1{background:color-mix(in srgb,#3b82f6 18%,var(--bg));color:var(--b)}
.s2{background:color-mix(in srgb,#10b981 20%,var(--bg));color:var(--g)}
.s3{background:color-mix(in srgb,#f59e0b 22%,var(--bg));color:var(--a)}
html.dark .s3{color:#fbbf24}
.s4{background:color-mix(in srgb,#ef4444 18%,var(--bg));color:var(--r)}
.s0{background:var(--soft)}
.seg{display:inline-flex;gap:2px;vertical-align:middle;margin-right:7px}
.seg i{width:5px;height:13px;border-radius:2px;background:var(--line)}
.seg i.on{background:var(--accent)}
.seg.full i.on{background:var(--g)}
.ttn{font-weight:600}.full+.ttn{color:var(--g)}
.up{color:var(--g);font-weight:600}.dn{color:var(--r);font-weight:600}
.faint{opacity:.35}
.ok{color:var(--g);font-weight:700}
.dotx{display:inline-block;width:10px;height:10px;border-radius:50%;background:var(--accent);
box-shadow:0 0 0 3px color-mix(in srgb,var(--accent) 25%,transparent)}
.bar{display:flex;align-items:center;gap:9px;min-width:150px}
.bar .t{flex:1;height:8px;border-radius:99px;background:var(--soft);overflow:hidden}
.bar .f{height:100%;border-radius:99px;background:linear-gradient(90deg,var(--accent),var(--accent2))}
.bar b{min-width:24px;text-align:right}
.txt{color:color-mix(in srgb,var(--fg) 85%,var(--bg))}
.empty{text-align:center;padding:36px 12px;opacity:.6}
@media (max-width:760px){.p3{display:none}}
@media (max-width:520px){.p2{display:none}td,th{padding:0 9px}}
</style></head><body>
<div class="wrap"><table><thead id="h"></thead><tbody id="b"></tbody></table></div>
<script>
const P=__PAYLOAD__;
const esc=s=>String(s==null?'':s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fx=(v,d)=>v==null?'<span class="faint">–</span>':Number(v).toFixed(d);
const sg=(v,d,suf)=>v==null?'<span class="faint">–</span>':'<span class="'+(v>0?'up':v<0?'dn':'faint')+'">'+(v>0?'+':'')+Number(v).toFixed(d)+suf+'</span>';
let sk=P.sort[0],sd=P.sort[1];
const pc=c=>(c.pri==2?'p2':c.pri==3?'p3':'')+(c.left?' l':'');
function cell(c,r){
  const v=r[c.k],k='<td class="'+pc(c)+'">';
  switch(c.type){
    case 'sym':return k+'<a class="sym" target="_blank" rel="noopener" href="https://www.tradingview.com/chart/?symbol=NSE%3A'+encodeURIComponent(v)+'">'+esc(v)+'</a></td>';
    case 'txt':return k+'<span class="ell" title="'+esc(v)+'">'+esc(v)+'</span></td>';
    case 'text':return k+(v?'<span class="txt">'+esc(v)+'</span>':'<span class="faint">–</span>')+'</td>';
    case 'rs':{if(v==null)return k+'–</td>';const q=v>=90?'c1':v>=80?'c2':v>=70?'c3':v>=50?'c4':'c5';return k+'<span class="badge '+q+'">'+v+'</span></td>';}
    case 'chg':return k+(v>0?'<span class="up">▲ '+v+'</span>':v<0?'<span class="dn">▼ '+Math.abs(v)+'</span>':'<span class="faint">0</span>')+'</td>';
    case 'sgn1':return k+sg(v,1,'')+'</td>';
    case 'spct':return k+sg(v,1,'%')+'</td>';
    case 'stage':{if(!v||v==='n/a')return k+'<span class="chipx s0">n/a</span></td>';const n=(v.match(/\d/)||[0])[0];return k+'<span class="chipx s'+n+'">'+esc(v.replace('Stage ','').replace(' · ',' · '))+'</span></td>';}
    case 'tt':{let s='';for(let i=1;i<=8;i++)s+='<i class="'+(i<=v?'on':'')+'"></i>';return k+'<span class="seg'+(v==8?' full':'')+'">'+s+'</span><span class="ttn">'+v+'/8</span></td>';}
    case 'chk':return k+(v?'<span class="ok">✓</span>':'<span class="faint">·</span>')+'</td>';
    case 'dot':return k+(v?'<span class="dotx" title="RS line at 52-week high before price"></span>':'<span class="faint">·</span>')+'</td>';
    case 'n0':return k+(v==null?'–':Math.round(v))+'</td>';
    case 'n1':return k+fx(v,1)+'</td>';
    case 'n2':return k+fx(v,2)+'</td>';
    case 'pct':return k+(v==null?'<span class="faint">–</span>':Number(v).toFixed(1)+'%')+'</td>';
    case 'pct0':return k+(v==null?'–':Math.round(v)+'%')+'</td>';
    case 'days':return k+(v?v+'d':'<span class="faint">–</span>')+'</td>';
    case 'ago':return k+(v==null?'<span class="faint">–</span>':v==0?'<span class="up">today</span>':v+'d ago')+'</td>';
    case 'mult':return k+(v==null?'<span class="faint">–</span>':Number(v).toFixed(1)+'×')+'</td>';
    case 'bar':return k+'<div class="bar"><div class="t"><div class="f" style="width:'+Math.max(0,Math.min(99,v||0))+'%"></div></div><b>'+(v==null?'–':v)+'</b></div></td>';
  }
  return k+esc(v)+'</td>';
}
function draw(){
  const rows=P.rows.slice();
  rows.sort((a,b)=>{let x=a[sk],y=b[sk];
    if(x==null&&y==null)return 0;if(x==null)return 1;if(y==null)return -1;
    if(typeof x==='string')return sd*x.localeCompare(y);
    return sd*(Number(x)-Number(y));});
  document.getElementById('h').innerHTML='<tr>'+P.cols.map(c=>'<th class="'+pc(c)+'" data-k="'+c.k+'">'+esc(c.t)+'<span class="ar">'+(c.k===sk?(sd>0?'▲':'▼'):'')+'</span></th>').join('')+'</tr>';
  document.getElementById('b').innerHTML=rows.length?rows.map(r=>'<tr>'+P.cols.map(c=>cell(c,r)).join('')+'</tr>').join('')
    :'<tr><td class="empty" colspan="'+P.cols.length+'" style="text-align:center;position:static">'+esc(P.empty)+'</td></tr>';
  document.querySelectorAll('th').forEach(th=>th.onclick=()=>{const k=th.dataset.k;
    if(k===sk)sd=-sd;else{sk=k;sd=P.cols.find(c=>c.k===k).left?1:-1;}draw();});
}
// follow the Streamlit theme (light / dark / custom) live
function sync(){
  let bg,fg;
  try{const app=window.parent.document.querySelector('.stApp'),cs=window.parent.getComputedStyle(app);bg=cs.backgroundColor;fg=cs.color;}catch(e){}
  if(!bg||bg==='rgba(0, 0, 0, 0)'||!fg){const d=matchMedia('(prefers-color-scheme: dark)').matches;bg=d?'rgb(14,17,23)':'rgb(255,255,255)';fg=d?'rgb(250,250,250)':'rgb(49,51,63)';}
  const r=document.documentElement;r.style.setProperty('--bg',bg);r.style.setProperty('--fg',fg);
  const m=(bg.match(/[\d.]+/g)||[255,255,255]).map(Number);r.classList.toggle('dark',(0.299*m[0]+0.587*m[1]+0.114*m[2])<128);
}
sync();setInterval(sync,1000);draw();
</script></body></html>"""


def table(df, keys, sort, font="Inter", density="Comfortable",
          empty="No stocks match the current filters."):
    """Render a sortable, theme-aware table. `sort` = (column key, -1 desc | 1 asc)."""
    cols = [dict(SPEC[k], k=k) for k in keys if k in df.columns and k in SPEC]
    data = json.loads(df[[c["k"] for c in cols]].to_json(orient="records")) if len(df) else []
    rowh = 31 if density == "Compact" else 41
    height = (120 if not data else min(len(data) * rowh + 46, 640)) + 4
    payload = json.dumps({"cols": cols, "rows": data, "sort": list(sort), "empty": empty}).replace("</", "<\\/")
    html = (_TABLE_HTML
            .replace("__IMPORT__", f"@import url('{font_url(font)}');" if font_url(font) else "")
            .replace("__FAMILY__", font_family(font))
            .replace("__ROWH__", str(rowh))
            .replace("__PAYLOAD__", payload))
    _embed(html, height)


def _embed(html: str, height: int) -> None:
    """st.iframe on current Streamlit; the older components.html call as a fallback."""
    if hasattr(st, "iframe"):
        st.iframe(html, height=height)
    else:
        components.html(html, height=height, scrolling=False)
