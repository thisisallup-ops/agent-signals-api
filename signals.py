"""
The actual product: your market signals.

This is where you plug in the logic you already built in daily_outlook.py.
The two functions below are the only contract main.py cares about — each
returns a plain dict that becomes JSON.

Right now they return realistic placeholder data so you can test the payment
flow end-to-end TODAY. Then replace the placeholder blocks with your real
yfinance code (Asian overnight consensus, seasonality tables, etc.).
"""

from datetime import datetime, timedelta, timezone


def _compute_signals(for_date: datetime) -> dict:
    """Replace the body of this function with your daily_outlook.py logic.

    Tips for porting:
      - Your GitHub Actions script builds an HTML email. Here you want the
        same numbers as a dict, BEFORE they get formatted into HTML.
      - yfinance calls can be slow (~seconds). That's fine for now; later
        you can cache the result once per morning so every paid request
        after the first is instant.
    """
    # ----------------- PLACEHOLDER — replace with your code -----------------
    return {
        "date": for_date.strftime("%Y-%m-%d"),
        "asian_overnight_consensus": {
            "kospi": "up",
            "nikkei": "up",
            "hang_seng": "up",
            "unanimous": True,
            "signal": "bullish",
            "historical_hit_rate": {"QQQ": 0.62, "SPY": 0.60, "IWM": 0.58},
            "note": "Hit rates from walk-forward test 2021-2026; "
            "no edge found for DIA.",
        },
        "seasonality": {
            "day_of_week": for_date.strftime("%A"),
            "holiday_drift_active": False,
        },
        "disclaimer": "Historical statistics, not financial advice. "
        "Past performance does not guarantee future results.",
    }
    # ------------------------------------------------------------------------


def get_todays_signals() -> dict:
    """What paying agents receive."""
    now = datetime.now(timezone.utc)
    data = _compute_signals(now)
    data["generated_at"] = now.isoformat()
    return data


def get_free_sample() -> dict:
    """Yesterday's signal, given away free on the landing endpoint.

    Free samples are how agents (and the humans configuring them) decide
    your data is worth paying for. Yesterday's signal has no trading value
    but fully demonstrates the format and content.
    """
    yesterday = datetime.now(timezone.utc) - timedelta(days=1)
    data = _compute_signals(yesterday)
    data["note"] = "FREE SAMPLE (yesterday's signal). Pay /signals for today's."
    return data
