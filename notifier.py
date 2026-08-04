"""
telegram/notifier.py — same pattern as the execution bot's notifier.
Silently no-ops if not configured (alerting is optional, not required).
"""
import os
import logging
import requests

log = logging.getLogger("research_notifier")

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")


def is_configured():
    return bool(BOT_TOKEN and CHAT_ID)


def send_message(text):
    if not is_configured():
        log.info(f"[Telegram not configured — message not sent]\n{text}")
        return
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        # Telegram caps messages at 4096 chars; split if needed
        for i in range(0, len(text), 4000):
            requests.post(url, json={"chat_id": CHAT_ID, "text": text[i:i+4000]}, timeout=10)
    except Exception as e:
        log.warning(f"Telegram send failed: {e}")
