# Nifty Total Market RS Scanner

A Streamlit scanner for Nifty Total Market constituents. The official index combines Nifty 500 and Nifty Microcap 250 constituents.

## Features

- RS ratings, Minervini Trend Template, VCP setups, volume breakouts and Weinstein stages across the full available constituent list.
- Per-stock 20-session average traded value (close × volume) and a liquid flag. A stock is liquid at ₹10 crore or more; the optional sidebar filter can limit stock screens to liquid names.
- Optional market-cap filters for stocks above ₹500 crore or ₹1,000 crore. Market caps load from Yahoo Finance only when a threshold is selected and are cached for 24 hours.
- Basic-industry strength and market breadth summaries calculated from liquid stocks only.
- Breadth percentages for liquid stocks above their 50, 150 and 200-day averages, plus Trend Template and Stage 2 counts.
- Stock detail candlesticks with EMA 10/21, SMA 50/200, volume with its 50-session average, and a relative-strength panel against Nifty 500.
- Historical consolidation bases are drawn as shaded rectangles. A newest-first 12-month base-history table reports start, length, depth, pivot, breakout, breakout volume and outcome; pivot and 52-week-high levels can be toggled.
- Search, industry filters, downloadable CSVs, cached data and manual refresh.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this project folder to a GitHub repository.
2. Create an app from that repository and select `app.py` as the main file.
3. Deploy. The app downloads constituents and daily prices when it runs; it does not require API secrets.

If the constituent download is unavailable, upload `ind_niftytotalmarket_list.csv` from Nifty Indices. The file needs `Company Name`, `Symbol`, and `Industry` or `Basic Industry` columns. A `Sector` column is used for sector breadth when present; otherwise breadth is grouped by basic industry.

## Method notes

- The liquidity value is the mean daily close × volume over the latest 20 sessions. Fewer than 20 complete sessions does not qualify as liquid.
- RS rankings and individual stock screens use the full rated Nifty Total Market universe unless “Only show liquid stocks” is selected.
- Industry strength and market breadth exclude stocks that fail the liquidity rule.
- The index constituent file and Yahoo Finance histories may update at different times or have missing prices and volume. Pattern labels are screening aids, not investment advice.
