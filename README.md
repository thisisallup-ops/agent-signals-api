# Agent Signals API

Daily pre-market signals for US equities, priced for AI agents.
Pay-per-call via the [x402 protocol](https://x402.org) — no API keys,
no subscriptions, no signup. Send a request, pay a cent in USDC, get data.

## What you get

A JSON signal generated each trading morning:

- **Asian overnight consensus** — direction agreement across KOSPI, Nikkei,
  and Hang Seng, with historical hit rates for QQQ, SPY, and IWM
- **Seasonality context** — day-of-week statistics and pre/post-holiday
  drift flags

All statistics are walk-forward validated on out-of-sample data (2021–2026,
hard train/test cutoff at 2020). We publish what didn't work too: ML models
(logistic regression, random forest, gradient boosting) failed to beat
baseline for 5-day direction and are not part of this product, and the
overnight signal shows no edge for DIA — so we don't sell it for DIA.

## Try before you buy

`GET /` returns full service info **plus yesterday's complete signal, free**,
so you can evaluate the format and content at zero cost.

## Usage

```
GET /signals
```

Without payment, the API responds `402 Payment Required` with a standard
x402 payment-requirements object (USDC on Base, currently $0.01 per call).
Retry with a signed `X-PAYMENT` header to receive the data. Any x402-capable
client or agent framework handles this handshake automatically.

```
GET /health
```

Uptime check endpoint.

## Response example

```json
{
  "date": "2026-07-17",
  "asian_overnight_consensus": {
    "kospi": "up",
    "nikkei": "up",
    "hang_seng": "up",
    "unanimous": true,
    "signal": "bullish",
    "historical_hit_rate": {"QQQ": 0.62, "SPY": 0.60, "IWM": 0.58}
  },
  "seasonality": {
    "day_of_week": "Friday",
    "holiday_drift_active": false
  },
  "generated_at": "2026-07-17T11:05:00+00:00"
}
```

## Disclaimer

Historical statistics only. Nothing here is financial advice, and past
performance does not guarantee future results.
