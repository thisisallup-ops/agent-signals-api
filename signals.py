from datetime import datetime, timedelta, timezone
import datetime as dt
import json
import sys
import numpy as np
import pandas as pd
import yfinance as yf

"""
Daily Market Outlook Script
============================
Consolidates the validated signals built in the research notebook:
  1. Historical calendar-date baseline (day-of-year performance)
  2. Holiday-anchor pattern (for dates near market holidays)
  3. Overnight Asia market signal (KOSPI/Nikkei/Hang Seng unanimous agreement)
Run this daily (ideally before US market open) to get a fresh outlook.
Designed to be scheduled via GitHub Actions (see accompanying workflow file).
IMPORTANT - what this script does NOT do:
  - It does not predict prices with certainty. All output is historical
    frequency / conditional probability, honestly labeled with sample sizes.
  - The overnight signal only applies to SAME-DAY direction, not multi-day
    forecasts.
  - Small sample sizes (shown as "N=") should be weighted accordingly -
    N < 30 is not statistically reliable.
"""

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
FULL_WATCHLIST = [
    "IREN", "NVDA", "AMD", "AVGO", "TSM", "MU", "ANET", "CRWV", "NBIS", "MSTR",
    "SPY", "QQQ", "DIA", "IWM",
    "NQ=F", "ES=F", "RTY=F", "YM=F",
    "^GSPC", "^IXIC", "^DJI", "^RUT", "^SOX", "^VIX",
    "^IRX", "^FVX", "^TNX", "^TYX",
    "CADUSD=X", "EURUSD=X", "JPY=X", "DX-Y.NYB",
    "CL=F", "GC=F", "SI=F", "HG=F", "NG=F",
    "BTC-USD", "ETH-USD", "SOL-USD",
    "^KS11", "^N225", "^HSI", "000001.SS",
    "BATL", "INDO",
]
PRIMARY_SYMBOLS = ["QQQ", "SPY", "DIA", "IWM"]
ASIA_SIGNAL_SYMBOLS = ["^KS11", "^N225", "^HSI"]
ALL_SYMBOLS = sorted(set(FULL_WATCHLIST) | set(ASIA_SIGNAL_SYMBOLS))
LOOKBACK_START = "1995-01-01"
MIN_YEARS_FOR_STATS = 5

# ---------------------------------------------------------------------------
# YOUR ORIGINAL FUNCTIONS
# ---------------------------------------------------------------------------
def load_price_data(symbols, start=LOOKBACK_START):
    print(f"Downloading data for {len(symbols)} symbols since {start}...", file=sys.stderr)
    raw = yf.download(symbols, start=start, group_by="ticker", auto_adjust=True, threads=True)
    frames = {}
    for sym in symbols:
        try:
            df = raw[sym].copy() if len(symbols) > 1 else raw.copy()
            frames[sym] = df["Close"]
        except (KeyError, TypeError):
            print(f" WARNING: no data returned for {sym}", file=sys.stderr)
    wide_close = pd.DataFrame(frames)
    wide_close = wide_close.sort_index()
    return wide_close

def day_of_year_stats(wide_close, target_symbol, month, day):
    returns = wide_close[target_symbol].pct_change(1, fill_method=None).dropna()
    df = pd.DataFrame({"return": returns})
    df["month"] = df.index.month
    df["day"] = df.index.day
    rows = df[(df["month"] == month) & (df["day"] == day)]
    if len(rows) == 0:
        return {"n_years": 0, "pct_positive": None, "avg_return": None, "note": "no trading history (likely holiday)"}
    return {
        "n_years": int(len(rows)),
        "pct_positive": round(float((rows["return"] > 0).mean()), 4),
        "avg_return": round(float(rows["return"].mean()), 5),
        "times_up": int((rows["return"] > 0).sum()),
    }

def trading_day_offsets(wide_close, anchor_dates, symbol, window=10):
    prices = wide_close[symbol].dropna()
    returns = prices.pct_change(fill_method=None)
    trading_days = prices.index
    records = []
    for anchor in anchor_dates:
        anchor = pd.Timestamp(anchor)
        idx_candidates = trading_days[trading_days >= anchor]
        if len(idx_candidates) == 0:
            continue
        matched_date = idx_candidates[0]
        if (matched_date - anchor).days > 10:
            continue
        anchor_idx = trading_days.get_loc(matched_date)
        for offset in range(-window, window + 1):
            pos = anchor_idx + offset
            if 0 <= pos < len(trading_days):
                date = trading_days[pos]
                if date in returns.index and not pd.isna(returns.loc[date]):
                    records.append({"year": anchor.year, "offset": offset, "return": returns.loc[date]})
    return pd.DataFrame(records)

def holiday_window_stats(wide_close, target_symbol, anchor_month_day, offset_start, offset_end, years=range(1998, 2027)):
    anchors = [f"{y}-{anchor_month_day}" for y in years]
    df = trading_day_offsets(wide_close, anchors, target_symbol, window=max(abs(offset_start), abs(offset_end)) + 1)
    window = df[(df["offset"] >= offset_start) & (df["offset"] <= offset_end)]
    yearly = window.groupby("year")["return"].apply(lambda x: (1 + x).prod() - 1)
    if len(yearly) == 0:
        return {"n_years": 0, "pct_positive": None, "avg_return": None}
    return {
        "n_years": int(len(yearly)),
        "pct_positive": round(float((yearly > 0).mean()), 4),
        "avg_return": round(float(yearly.mean()), 5),
        "times_up": int((yearly > 0).sum()),
    }

HOLIDAY_ANCHORS = {
    "New Year": "01-01",
    "Independence Day": "07-04",
    "Thanksgiving-adjacent (Nov 28 approx)": "11-28",
    "Christmas/Year-end": "12-25",
}

def nearest_holiday_context(wide_close, target_symbol, ref_date):
    ref = pd.Timestamp(ref_date)
    for name, month_day in HOLIDAY_ANCHORS.items():
        anchor = pd.Timestamp(f"{ref.year}-{month_day}")
        delta_days = (ref - anchor).days
        if abs(delta_days) <= 7:
            pre = holiday_window_stats(wide_close, target_symbol, month_day, -4, -1)
            post = holiday_window_stats(wide_close, target_symbol, month_day, 1, 5)
            return {"holiday": name, "anchor_date": str(anchor.date()), "pre_holiday": pre, "post_holiday": post}
    return None

def week_by_daterange_stats(wide_close, target_symbol, ref_date, years=range(1998, 2027)):
    ref_date = pd.Timestamp(ref_date)
    monday = ref_date - pd.Timedelta(days=ref_date.weekday())
    friday = monday + pd.Timedelta(days=4)
    if target_symbol not in wide_close.columns:
        return {"n_years": 0, "pct_positive": None, "avg_return": None}
    prices = wide_close[target_symbol].dropna()
    results = []
    for yr in years:
        try:
            yr_monday = monday.replace(year=yr)
            yr_friday = friday.replace(year=yr)
        except ValueError:
            continue
        window = prices[(prices.index >= yr_monday) & (prices.index <= yr_friday)]
        if len(window) >= 2:
            week_return = (window.iloc[-1] / window.iloc[0]) - 1
            results.append(week_return)
    if len(results) == 0:
        return {"n_years": 0, "pct_positive": None, "avg_return": None, "week_range": f"{monday.strftime('%m/%d')}-{friday.strftime('%m/%d')}"}
    results = pd.Series(results)
    return {
        "n_years": int(len(results)),
        "pct_positive": round(float((results > 0).mean()), 4),
        "avg_return": round(float(results.mean()), 5),
        "times_up": int((results > 0).sum()),
        "week_range": f"{monday.strftime('%m/%d')}-{friday.strftime('%m/%d')}",
    }

VALIDATED_AGREE_CONFLICT_ACCURACY = {
    "QQQ": (0.565, 0.431, "~207-211"),
    "SPY": (0.527, 0.423, "~194-224"),
    "IWM": (0.537, 0.460, "~189-229"),
}

def net_read(day_stats, overnight_sig, symbol=None):
    hist_lean = None
    if day_stats.get("pct_positive") is not None:
        hist_lean = "UP" if day_stats["pct_positive"] > 0.5 else "DOWN"
    status = overnight_sig.get("status")
    if status == "signal_up":
        overnight_lean = "UP"
    elif status == "signal_down":
        overnight_lean = "DOWN"
    else:
        overnight_lean = None
    if hist_lean is None or overnight_lean is None:
        return "Only one signal available today (or neither) - no combined read, see individual sections above."
    if hist_lean == overnight_lean:
        base = f"Historical baseline AND overnight signal both lean {hist_lean} - signals AGREE."
        if symbol in VALIDATED_AGREE_CONFLICT_ACCURACY:
            acc, _, n = VALIDATED_AGREE_CONFLICT_ACCURACY[symbol]
            base += f" (Backtested: agreement has coincided with ~{acc:.0%} next-day accuracy, N={n} unseen test-period days.)"
        return base
    else:
        base = f"Historical baseline leans {hist_lean} but overnight signal leans {overnight_lean} - signals CONFLICT."
        if symbol in VALIDATED_AGREE_CONFLICT_ACCURACY:
            _, acc, n = VALIDATED_AGREE_CONFLICT_ACCURACY[symbol]
            base += f" (Backtested: conflicts like this have coincided with only ~{acc:.0%} accuracy, N={n} - treat with real caution.)"
        return base

def overnight_asia_signal(wide_close, target_symbol, as_of_date, asia_symbols=ASIA_SIGNAL_SYMBOLS):
    asia_moves_all = wide_close[asia_symbols].pct_change(1, fill_method=None)
    as_of_date = pd.Timestamp(as_of_date)
    if as_of_date not in asia_moves_all.index or asia_moves_all.loc[as_of_date].isna().any():
        return {"status": "no_data", "detail": "No overnight data available for this date"}
    day_moves = asia_moves_all.loc[as_of_date]
    n_up = int((day_moves > 0).sum())
    n_down = int((day_moves < 0).sum())
    n_total = len(asia_symbols)
    moves_dict = {sym: round(float(day_moves[sym]), 4) for sym in asia_symbols}
    if n_up == n_total:
        return {
            "status": "signal_up",
            "moves": moves_dict,
            "note": "All Asian markets up overnight. Validated historical edge: QQQ/SPY/DIA up ~61-70% of matching days (N~200, 2021-2026 test period).",
        }
    elif n_down == n_total:
        return {
            "status": "signal_down",
            "moves": moves_dict,
            "note": "All Asian markets down overnight. Validated historical edge: QQQ/SPY/DIA down ~56-58% of matching days (N~200, 2021-2026 test period).",
        }
    else:
        return {"status": "no_signal", "moves": moves_dict, "note": "Mixed overnight moves - no unanimous signal today."}

def build_outlook(wide_close, target_symbol, as_of_date):
    as_of_date = pd.Timestamp(as_of_date)
    result = {
        "symbol": target_symbol,
        "as_of_date": str(as_of_date.date()),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    day_stats = day_of_year_stats(wide_close, target_symbol, as_of_date.month, as_of_date.day)
    result["day_of_year_baseline"] = day_stats
    holiday_ctx = nearest_holiday_context(wide_close, target_symbol, as_of_date)
    result["holiday_context"] = holiday_ctx
    all_returns = wide_close[target_symbol].pct_change(1, fill_method=None).dropna()
    result["all_time_pct_positive"] = round(float((all_returns > 0).mean()), 4)
    overnight_sig = overnight_asia_signal(wide_close, target_symbol, as_of_date)
    result["overnight_signal"] = overnight_sig
    result["net_read"] = net_read(day_stats, overnight_sig, symbol=target_symbol)
    return result

def format_sections_ab(outlook):
    lines = []
    lines.append(f"=== {outlook['symbol']} - {outlook['as_of_date']} ===")
    lines.append("")
    lines.append("[A] HISTORICAL OUTLOOK")
    day_stats = outlook["day_of_year_baseline"]
    if day_stats["n_years"] >= MIN_YEARS_FOR_STATS:
        lines.append(f" This calendar date ({day_stats['n_years']} years): up {day_stats['times_up']}/{day_stats['n_years']} ({day_stats['pct_positive']:.1%}), avg {day_stats['avg_return']:+.3%}")
    else:
        lines.append(f" Insufficient same-date history ({day_stats['n_years']} years) - likely near a holiday.")
    if outlook["holiday_context"]:
        hc = outlook["holiday_context"]
        lines.append(f" Holiday context: near {hc['holiday']} ({hc['anchor_date']})")
        pre, post = hc["pre_holiday"], hc["post_holiday"]
        if pre["n_years"] > 0:
            lines.append(f" Pre-holiday (4 days before): up {pre['times_up']}/{pre['n_years']} ({pre['pct_positive']:.1%}), avg {pre['avg_return']:+.3%}")
        if post["n_years"] > 0:
            lines.append(f" Post-holiday (5 days after): up {post['times_up']}/{post['n_years']} ({post['pct_positive']:.1%}), avg {post['avg_return']:+.3%}")
    lines.append(f" (all-time average: {outlook['all_time_pct_positive']:.1%} of days positive)")
    lines.append("")
    lines.append("[B] OVERNIGHT SIGNAL & NET READ")
    sig = outlook["overnight_signal"]
    lines.append(f" {sig.get('note', sig.get('detail', 'unknown'))}")
    if "moves" in sig:
        moves_str = ", ".join(f"{k} {v:+.2%}" for k, v in sig["moves"].items())
        lines.append(f" ({moves_str})")
    lines.append(f" NET READ: {outlook['net_read']}")
    lines.append("")
    return "\n".join(lines)

def build_section_c_table(wide_close, as_of_date, symbols):
    rows = []
    for sym in symbols:
        if sym not in wide_close.columns:
            continue
        stats = week_by_daterange_stats(wide_close, sym, as_of_date)
        rows.append({
            "symbol": sym,
            "week_range": stats.get("week_range", ""),
            "n_years": stats["n_years"],
            "pct_positive": stats["pct_positive"],
            "avg_return": stats["avg_return"],
        })
    return rows

def format_section_c(rows):
    lines = []
    lines.append("[C] THIS WEEK vs HISTORY - ALL WATCHLIST SYMBOLS")
    if rows:
        lines.append(f" (calendar week {rows[0]['week_range']}, matched by date across available years)")
    lines.append(f" {'Symbol':<12}{'N yrs':>7}{'% up':>8}{'Avg return':>12}")
    lines.append(" " + "-" * 39)
    for r in rows:
        if r["n_years"] >= MIN_YEARS_FOR_STATS:
            lines.append(f" {r['symbol']:<12}{r['n_years']:>7}{r['pct_positive']:>8.1%}{r['avg_return']:>+12.3%}")
        else:
            lines.append(f" {r['symbol']:<12}{'(insufficient history - likely holiday-adjacent)':>39}")
    lines.append("")
    return "\n".join(lines)

WEEKDAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri"]

def day_of_week_distribution(wide_close, symbol):
    returns = wide_close[symbol].pct_change(1, fill_method=None).dropna()
    if len(returns) == 0:
        return None
    df = pd.DataFrame({"return": returns})
    df["weekday"] = df.index.weekday
    result = {"symbol": symbol, "n_obs": int(len(df))}
    for i, name in enumerate(WEEKDAY_NAMES):
        day_rows = df[df["weekday"] == i]
        if len(day_rows) >= MIN_YEARS_FOR_STATS:
            result[name] = round(float((day_rows["return"] > 0).mean()), 4)
        else:
            result[name] = None
    return result

def build_section_d_table(wide_close, symbols):
    rows = []
    for sym in symbols:
        if sym not in wide_close.columns:
            continue
        r = day_of_week_distribution(wide_close, sym)
        if r is not None:
            rows.append(r)
    return rows

def format_section_d(rows):
    lines = []
    lines.append("[D] HISTORICAL DAY-OF-WEEK DISTRIBUTION - ALL WATCHLIST SYMBOLS")
    lines.append(" (% of days positive, by weekday, using each symbol's full available history)")
    header = f" {'Symbol':<12}{'N obs':>7}"
    for name in WEEKDAY_NAMES:
        header += f"{name:>8}"
    lines.append(header)
    lines.append(" " + "-" * (12 + 7 + 8 * 5))
    for r in rows:
        line = f" {r['symbol']:<12}{r['n_obs']:>7}"
        for name in WEEKDAY_NAMES:
            val = r[name]
            line += f"{val:>7.1%} " if val is not None else f"{'n/a':>8}"
        lines.append(line)
    lines.append("")
    lines.append(" NOTE: this is total-history seasonality...")
    lines.append("")
    return "\n".join(lines)

def main():
    as_of_date = dt.datetime.today()
    wide_close = load_price_data(ALL_SYMBOLS)
    # ... (the rest of your main() function as you had it)
    # (I kept it short here for brevity - keep your full main() if you want the text output too)
    print("Signals computed for API")

# ---------------------------------------------------------------------------
# API INTERFACE
# ---------------------------------------------------------------------------
_cache = {"date": None, "data": None}

def _compute_signals(for_date: datetime) -> dict:
    global _cache
    today_str = for_date.date().isoformat()
    if _cache.get("date") == today_str and _cache.get("data"):
        return _cache["data"]
    try:
        wide_close = load_price_data(ALL_SYMBOLS)
        all_outlooks = [build_outlook(wide_close, sym, for_date) for sym in PRIMARY_SYMBOLS if sym in wide_close.columns]
        section_c_rows = build_section_c_table(wide_close, for_date, FULL_WATCHLIST)
        section_d_rows = build_section_d_table(wide_close, FULL_WATCHLIST)
        result = {
            "date": today_str,
            "primary_outlooks": all_outlooks,
            "section_c_weekly_table": section_c_rows,
            "section_d_weekday_distribution": section_d_rows,
            "disclaimer": "Historical statistics only. Not financial advice.",
            "generated_at_utc": datetime.now(timezone.utc).isoformat()
        }
        _cache = {"date": today_str, "data": result}
        return result
    except Exception as e:
        return {"error": str(e)}

def get_todays_signals() -> dict:
    return _compute_signals(datetime.now(timezone.utc))

def get_free_sample() -> dict:
    """Yesterday's signal (free)."""
    yesterday = datetime.now(timezone.utc) - timedelta(days=1)
    # Make sure yesterday is tz-aware
    if yesterday.tzinfo is None:
        yesterday = yesterday.replace(tzinfo=timezone.utc)
    data = _compute_signals(yesterday)
    data["note"] = "FREE SAMPLE (yesterday's signal). Pay /signals for today's."
    return data

if __name__ == "__main__":
    main()
