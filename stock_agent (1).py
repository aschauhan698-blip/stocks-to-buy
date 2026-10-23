"""
Daily Indian stock screener agent.

Filters (NSE equities):
  - Up > 15% over the last 7 trading sessions
  - RSI(14) > 50
  - Promoter holding > 30%
  - Market cap > Rs 50 Cr
Outputs the top 20 (by 7-session return) to Telegram (or console).

Setup:  pip install -r requirements.txt
Env:    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID   (optional; prints if missing)
Run:    python stock_agent.py            (use --force to run on market holidays)
"""
import io
import os
import re
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup

MIN_RETURN_PCT = 15.0
LOOKBACK_SESSIONS = 7
MIN_RSI = 50.0
MIN_PROMOTER_PCT = 30.0
MIN_MCAP_CR = 50.0
TOP_N = 20

IST = ZoneInfo("Asia/Kolkata")
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36"}


def get_universe() -> list[str]:
    """All NSE EQ-series symbols."""
    url = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = [c.strip() for c in df.columns]
    df = df[df["SERIES"].str.strip() == "EQ"]
    return df["SYMBOL"].str.strip().tolist()


def rsi_wilder(close: pd.Series, period: int = 14) -> float:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    rsi = 100 - 100 / (1 + rs)
    return float(rsi.iloc[-1]) if not rsi.empty else float("nan")


def price_screen(symbols: list[str]) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Step 1: cheap technical filter (7-session return and RSI) using bulk price data."""
    rows, last_date = [], None
    chunk = 150
    for i in range(0, len(symbols), chunk):
        batch = symbols[i:i + chunk]
        tickers = [f"{s}.NS" for s in batch]
        try:
            data = yf.download(tickers, period="6mo", interval="1d", group_by="ticker",
                               auto_adjust=True, threads=True, progress=False)
        except Exception as e:
            print(f"batch {i} failed: {e}", file=sys.stderr)
            continue
        for s, t in zip(batch, tickers):
            try:
                close = data[t]["Close"].dropna()
                if len(close) < 30:
                    continue
                last_date = close.index[-1] if last_date is None else max(last_date, close.index[-1])
                ret = (close.iloc[-1] / close.iloc[-1 - LOOKBACK_SESSIONS] - 1) * 100
                if ret <= MIN_RETURN_PCT:
                    continue
                rsi = rsi_wilder(close)
                if rsi > MIN_RSI:
                    rows.append({"symbol": s, "price": round(close.iloc[-1], 2),
                                 "ret_7d": round(ret, 2), "rsi": round(rsi, 1)})
            except Exception:
                continue
    return pd.DataFrame(rows), last_date


def fundamentals(symbol: str) -> tuple[float | None, float | None]:
    """Step 2: market cap (Rs Cr) and latest promoter holding (%) from Screener.in."""
    for path in (f"company/{symbol}/consolidated/", f"company/{symbol}/"):
        r = requests.get(f"https://www.screener.in/{path}", headers=HEADERS, timeout=30)
        if r.status_code != 200:
            continue
        soup = BeautifulSoup(r.text, "html.parser")

        mcap = None
        for li in soup.select("#top-ratios li"):
            name = li.select_one(".name")
            if name and "Market Cap" in name.get_text():
                num = li.select_one(".number")
                if num:
                    mcap = float(num.get_text().replace(",", "").strip())

        promoter = None
        shp = soup.select_one("#quarterly-shp") or soup.select_one("#shareholding")
        if shp:
            for tr in shp.select("tr"):
                label = tr.select_one("td")
                if label and label.get_text().strip().lower().startswith("promoters"):
                    cells = [td.get_text() for td in tr.select("td")[1:]]
                    vals = [re.sub(r"[^\d.]", "", c) for c in cells]
                    vals = [float(v) for v in vals if v]
                    if vals:
                        promoter = vals[-1]  # latest quarter
                    break
        if mcap is not None or promoter is not None:
            return mcap, promoter
    return None, None


def build_report() -> str:
    symbols = get_universe()
    cands, last_date = price_screen(symbols)
    if cands.empty:
        return "No stocks passed the price/RSI filters today."

    cands = cands.sort_values("ret_7d", ascending=False)
    results = []
    for _, row in cands.iterrows():
        mcap, promoter = fundamentals(row["symbol"])
        time.sleep(1.0)  # be polite to Screener.in
        if mcap is None or promoter is None:
            continue
        if mcap > MIN_MCAP_CR and promoter > MIN_PROMOTER_PCT:
            results.append({**row.to_dict(), "mcap": mcap, "promoter": promoter})
        if len(results) >= TOP_N:
            break

    date_str = last_date.strftime("%d %b %Y") if last_date is not None else "n/a"
    if not results:
        return f"Screen for {date_str}: no stocks met all four criteria."

    lines = [f"📈 Momentum screen — {date_str}",
             f">{MIN_RETURN_PCT:.0f}% in {LOOKBACK_SESSIONS} sessions | RSI>{MIN_RSI:.0f} | "
             f"Promoter>{MIN_PROMOTER_PCT:.0f}% | MCap>₹{MIN_MCAP_CR:.0f} Cr", ""]
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r['symbol']}  ₹{r['price']}  +{r['ret_7d']}%  "
                     f"RSI {r['rsi']}  Promoter {r['promoter']}%  MCap ₹{r['mcap']:,.0f} Cr")
    lines += ["", "Screening output only, not investment advice."]
    return "\n".join(lines)


WHATSAPP_TO = os.getenv("WHATSAPP_TO", "919731899447")  # country code + number, no "+"


def chunk_lines(text: str, limit: int) -> list[str]:
    chunks, cur = [], ""
    for line in text.split("\n"):
        if len(cur) + len(line) + 1 > limit and cur:
            chunks.append(cur)
            cur = ""
        cur += line + "\n"
    if cur.strip():
        chunks.append(cur)
    return chunks


def send_whatsapp_twilio(text: str) -> bool:
    sid, tok = os.getenv("TWILIO_ACCOUNT_SID"), os.getenv("TWILIO_AUTH_TOKEN")
    if not (sid and tok):
        return False
    frm = os.getenv("TWILIO_WHATSAPP_FROM", "+14155238886")  # Twilio sandbox default
    for part in chunk_lines(text, 1500):
        requests.post(f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json",
                      auth=(sid, tok), timeout=30,
                      data={"From": f"whatsapp:{frm}", "To": f"whatsapp:+{WHATSAPP_TO}",
                            "Body": part}).raise_for_status()
    return True


def send_whatsapp_callmebot(text: str) -> bool:
    key = os.getenv("CALLMEBOT_APIKEY")
    if not key:
        return False
    for part in chunk_lines(text, 1000):
        requests.get("https://api.callmebot.com/whatsapp.php", timeout=30,
                     params={"phone": WHATSAPP_TO, "text": part, "apikey": key}).raise_for_status()
        time.sleep(2)
    return True


def send(text: str) -> None:
    sent = send_whatsapp_twilio(text) or send_whatsapp_callmebot(text)
    token, chat = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if token and chat:
        for i in range(0, len(text), 4000):
            requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                          json={"chat_id": chat, "text": text[i:i + 4000]}, timeout=30).raise_for_status()
        sent = True
    if not sent:
        print(text)


if __name__ == "__main__":
    report = build_report()
    send(report)
