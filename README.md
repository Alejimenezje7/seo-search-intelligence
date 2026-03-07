# Adidas Search Intelligence Platform

A lightweight Streamlit dashboard for monitoring Google Search Console trends across Adidas LatAm markets.

## Project structure

```
adidas-search-intelligence/
├── app.py                  # Streamlit entry point
├── config.py               # Domains, thresholds, settings
├── requirements.txt
├── .env.example
├── .gitignore
│
├── src/
│   ├── extractor.py        # GSC API extraction (auth, pagination, cleaning)
│   ├── cache.py            # Parquet-based local storage
│   ├── processor.py        # WoW and MTD comparison logic
│   ├── anomaly.py          # Z-score anomaly detection
│   └── filters.py          # Sidebar filter widgets
│
├── views/
│   ├── overview.py         # Dashboard homepage
│   ├── weekly.py           # Week-over-Week deep-dive
│   ├── mtd.py              # Month-to-Date analysis
│   └── explorer.py         # Free-form keyword explorer
│
└── data/
    └── raw/                # Parquet cache files (gitignored)
```

## Setup

```bash
# 1. Clone and enter the project
cd adidas-search-intelligence

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure credentials
cp .env.example .env
# Edit .env and set GSC_CREDENTIALS_FILE to the path of your service account JSON

# 4. Run
streamlit run app.py
```

## How data flows

1. Click **Refresh from API** in the sidebar.
2. The app calls `src/extractor.py` → pulls 5 weeks from all 8 domains.
3. Data is saved as a Parquet file in `data/raw/`.
4. On every subsequent load the app reads from disk (fast, no API call).
5. Run the refresh weekly (manually or via a scheduler) to keep data current.

## Comparison logic

| Mode | Period A | Period B |
|------|----------|----------|
| Week-over-Week | Last full Mon–Sun | The Mon–Sun before that |
| Month-to-Date | 1st of current month → yesterday | Same day-count window in prior month |

Minimum volume thresholds (configurable in `config.py`):
- `MIN_CLICKS_THRESHOLD = 10`
- `MIN_IMPRESSIONS_THRESHOLD = 50`

## Anomaly detection

Uses Z-score on the distribution of metric deltas across all keywords.
A keyword is flagged when:
- `|z_score| >= ANOMALY_ZSCORE_THRESHOLD` (default 2.0)
- AND the absolute delta exceeds the minimum meaningful size

## Adding a new market

In `config.py`:
```python
DOMAINS = [
    ...
    "https://www.adidas.com.uy/",   # Add here
]
DOMAIN_LABELS = {
    ...
    "https://www.adidas.com.uy/": "Uruguay",
}
```

## Planned next steps

- Scheduled weekly extraction via GitHub Actions or a cron job
- Page-level analysis (add `page` to GSC dimensions)
- Slack / email digest for anomaly alerts
- BigQuery backend for multi-user, long-term history
