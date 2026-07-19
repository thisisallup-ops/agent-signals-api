"""
Agent Signals API — a market-signals service that AI agents can pay to use.

How it works (the x402 flow):
  1. An agent calls GET /signals with no payment -> we reply 402 Payment Required
     with a machine-readable "price tag" (PaymentRequirements).
  2. The agent's wallet signs a USDC payment and retries the request with an
     X-PAYMENT header.
  3. We ask the facilitator (a free public verification server) to verify and
     settle the payment on-chain, then return the data.

You never touch crypto code directly — the facilitator does the hard part.

Run locally:   uvicorn main:app --reload
Deploy:        see README.md (Render free tier works)
"""

import base64
import json
import os
from datetime import datetime, timezone

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from x402.http import HTTPFacilitatorClientSync, DEFAULT_FACILITATOR_URL
from x402.schemas import (
    PaymentPayloadV1,
    PaymentRequiredV1,
    PaymentRequirementsV1,
)

from signals import get_free_sample, get_todays_signals

# ---------------------------------------------------------------------------
# Configuration — set these as environment variables on your host
# ---------------------------------------------------------------------------

# The wallet address that receives payments (YOUR wallet on Base).
# Create one free with Coinbase Wallet or MetaMask, switch network to Base.
PAY_TO_ADDRESS = os.environ.get("PAY_TO_ADDRESS", "0xYOUR_WALLET_ADDRESS_HERE")

# Price per request, in USDC "atomic units" (USDC has 6 decimals).
# "10000" = $0.01.  "1000" = $0.001.  Start cheap; you can raise it later.
PRICE_ATOMIC = os.environ.get("PRICE_ATOMIC", "10000")

# Network: "base-sepolia" = free test network (start here!)
#          "base"         = real money (switch after everything works)
NETWORK = os.environ.get("X402_NETWORK", "base-sepolia")

# USDC contract addresses (these are fixed, don't change them)
USDC_ADDRESS = {
    "base-sepolia": "0x036CbD53842c5426634e7929541eC2318f3dCF7e",
    "base": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
}[NETWORK]

# The facilitator verifies payments for you. x402.org's is free for testnet;
# for mainnet Base, Coinbase's CDP facilitator is the standard choice.
FACILITATOR_URL = os.environ.get("FACILITATOR_URL", DEFAULT_FACILITATOR_URL)

# Public URL of your deployed API (used in the price tag). Set after deploy.
BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000")

# ---------------------------------------------------------------------------

app = FastAPI(
    title="Agent Signals API",
    description="Daily market signals (Asian overnight consensus, seasonality) "
    "priced for AI agents via the x402 payment protocol.",
)

facilitator = HTTPFacilitatorClientSync({"url": FACILITATOR_URL})


def build_payment_requirements() -> PaymentRequirementsV1:
    """The machine-readable price tag we attach to 402 responses."""
    return PaymentRequirementsV1(
        scheme="exact",
        network=NETWORK,
        max_amount_required=PRICE_ATOMIC,
        resource=f"{BASE_URL}/signals",
        description="Today's market signals: Asian overnight consensus "
        "(KOSPI/Nikkei/Hang Seng) plus seasonality stats for QQQ/SPY/IWM.",
        mime_type="application/json",
        pay_to=PAY_TO_ADDRESS,
        max_timeout_seconds=120,
        asset=USDC_ADDRESS,
    )


@app.get("/")
def root():
    """Free endpoint: tells agents (and humans) what this service sells.

    Includes yesterday's signal as a free sample so buyers can judge quality
    before paying — this matters a lot for discovery.
    """
    return {
        "service": "Agent Signals API",
        "what_you_get": "Daily pre-market signal JSON: Asian overnight "
        "consensus direction + historical hit-rate context for QQQ/SPY/IWM.",
        "price_usd": int(PRICE_ATOMIC) / 1_000_000,
        "payment": "x402 (USDC on Base). Call /signals to receive a 402 "
        "response with payment requirements.",
        "network": NETWORK,
        "free_sample": get_free_sample(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/signals")
def signals(request: Request):
    """Paid endpoint. Implements the x402 handshake."""
    requirements = build_payment_requirements()

    payment_header = request.headers.get("X-PAYMENT")

    # --- Step 1: no payment attached -> send the price tag (HTTP 402) ------
    if not payment_header:
        body = PaymentRequiredV1(
            x402_version=1,
            error="Payment required",
            accepts=[requirements],
        )
        return JSONResponse(status_code=402, content=body.model_dump(by_alias=True))

    # --- Step 2: payment attached -> verify it with the facilitator --------
    try:
        payload_json = json.loads(base64.b64decode(payment_header))
        payload = PaymentPayloadV1.model_validate(payload_json)
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"error": "X-PAYMENT header is not a valid x402 payload"},
        )

    verify_result = facilitator.verify(payload, requirements)
    if not getattr(verify_result, "is_valid", False):
        return JSONResponse(
            status_code=402,
            content={
                "error": f"Payment invalid: {getattr(verify_result, 'invalid_reason', 'unknown')}",
                "accepts": [requirements.model_dump(by_alias=True)],
            },
        )

    # --- Step 3: settle (actually move the money), then serve the data -----
    settle_result = facilitator.settle(payload, requirements)
    if not getattr(settle_result, "success", False):
        return JSONResponse(
            status_code=402,
            content={"error": "Payment settlement failed, you were not charged"},
        )

    response = JSONResponse(content=get_todays_signals())
    # Receipt header so the buying agent can log the transaction
    response.headers["X-PAYMENT-RESPONSE"] = base64.b64encode(
        json.dumps(settle_result.model_dump(by_alias=True), default=str).encode()
    ).decode()
    return response


@app.get("/health")
def health():
    """For your hosting provider's uptime checks."""
    return {"ok": True}
