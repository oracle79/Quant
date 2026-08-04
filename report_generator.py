"""
reports/report_generator.py — ranks scanned markets by edge, confidence,
liquidity, and expected value, per spec. Formats the morning Telegram report.
"""

def rank_opportunities(scanned_markets):
    """scanned_markets: list of dicts with question, asset, timeframe,
    market_prob, model_prob, edge, confidence, liquidity, top_drivers."""
    def ev_score(m):
        # simple expected-value-style ranking: bigger edge, higher confidence,
        # and enough liquidity to actually matter, all multiply together
        liquidity_factor = min(1.0, (m.get("liquidity", 0) or 0) / 5000)
        return abs(m["edge"]) * m["confidence"] * liquidity_factor

    return sorted(scanned_markets, key=ev_score, reverse=True)


def format_morning_report(scanned_markets, top_n=10):
    ranked = rank_opportunities(scanned_markets)
    lines = [
        "📊 BTC/ETH Quant Research — Morning Report",
        f"Markets scanned: {len(scanned_markets)}",
        f"Potential opportunities (|edge| > 3pp): {sum(1 for m in scanned_markets if abs(m['edge'])>0.03)}",
        "",
    ]
    for m in ranked[:top_n]:
        direction = "UP" if m["edge"] > 0 else "DOWN"
        lines.append(
            f"• {m['asset']} {m['timeframe']}: mkt {m['market_prob']*100:.1f}% | "
            f"model {m['model_prob']*100:.1f}% | edge {m['edge']*100:+.1f}pp ({direction}) | "
            f"conf {m['confidence']*100:.0f}%"
        )
        for name, expl in m.get("top_drivers", [])[:2]:
            lines.append(f"    - {name}: {expl}")
    if not ranked:
        lines.append("No markets scanned this cycle.")
    return "\n".join(lines)
