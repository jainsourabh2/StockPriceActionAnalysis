# Financial Agent for Indian Markets

A hands-on project showing how to build an AI-powered financial analysis agent for Indian equities using [AngelOne's SmartAPI](https://smartapi.angelbroking.com/) and [Claude](https://docs.anthropic.com/en/docs/overview) (Anthropic's API).

You feed it a stock ticker. It pulls live market data from NSE/BSE, runs it through Claude, and gives you a structured analysis — trend direction, support/resistance levels, volume analysis, and a plain English summary. There's also an enhanced version that cross-references options chain data (IV, OI, Greeks) against price action.

**This is not a trading bot.** It's a starting point to show what's possible when you combine a free brokerage API with an LLM. Think of it as building blocks for something bigger.

---

## Why AngelOne SmartAPI?

Most Indian market data APIs either charge a monthly fee or lock useful features behind paid tiers. AngelOne's SmartAPI is genuinely free — you just need a trading account (which you probably already have if you trade Indian markets).

What you get for free:
- **Live NSE/BSE quotes** — LTP, OHLC, bid/ask
- **Historical candles** — 1min to daily, going back years
- **Full option chains** — with Greeks (delta, gamma, theta, vega) and IV
- **WebSocket streaming** — real-time tick data
- **Order placement** — buy, sell, modify, cancel via API
- **Portfolio & positions** — current holdings and P&L

No rate limit gotchas for normal usage. No surprise bills. It just works.

---

## What's in this repo

| File | What it does |
|---|---|
| `connect.py` | Authenticates with SmartAPI, gives you a session |
| `fetch_data.py` | Pulls 30 days of daily OHLCV candles for any stock |
| `agent.py` | The main agent — feeds price data to Claude, gets structured analysis |
| `agent_with_options.py` | Enhanced agent that also pulls option chain data for deeper analysis |
| `place_order.py` | Reference script for placing orders (dry-run by default, won't execute) |
| `config.py` | Loads credentials from `.env` |

---

## Prerequisites

- Python 3.8+
- An AngelOne trading account
- An Anthropic API key (for Claude)

---

## Setup

### 1. Get your AngelOne SmartAPI credentials

1. Go to [smartapi.angelbroking.com](https://smartapi.angelbroking.com/)
2. Sign up / log in with your AngelOne client ID (the one you use to trade)
3. Go to **My Apps** → **Create an App**
   - App name: anything you want (e.g., "financial-agent")
   - Redirect URL: `https://localhost` (doesn't matter for API usage)
4. You'll get an **API Key** — save it
5. **Enable TOTP** on your AngelOne account:
   - Open the AngelOne mobile app → Settings → Security → Enable TOTP
   - You'll get a TOTP secret (a long base32 string) — **save this**, you'll need it for automated login
   - If you already use Google Authenticator or similar, the secret is what you scanned as a QR code. You may need to re-enable TOTP to see the raw secret.

You'll need these four things:
- API Key (from SmartAPI dashboard)
- Client ID (your trading login ID, like `A12345678`)
- Password (your trading PIN)
- TOTP Secret (the base32 string, NOT the 6-digit code)

### 2. Get your Anthropic API key

1. Go to [console.anthropic.com](https://console.anthropic.com/)
2. Sign up or log in
3. Go to **API Keys** → **Create Key**
4. Save the key

### 3. Clone and install

```bash
git clone https://github.com/AM1403x/financial-agent-india.git
cd financial-agent-india

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 4. Configure credentials

```bash
cp .env.example .env
```

Open `.env` and fill in your actual values:

```
ANGELONE_API_KEY=your_smartapi_key_here
ANGELONE_CLIENT_ID=A12345678
ANGELONE_PASSWORD=1234
ANGELONE_TOTP_SECRET=your_totp_base32_secret
ANTHROPIC_API_KEY=sk-ant-...
```

### 5. Test the connection

```bash
python connect.py
```

If everything is set up correctly:
```
Generating TOTP...
Authenticating with AngelOne SmartAPI...
Login successful!
Session token: eyJhbGci...
Feed token: abc123...
Connected as client: A12345678
```

If it fails, you'll get a clear error message telling you what went wrong.

---

## Usage

### Fetch stock data

```bash
# Default: RELIANCE
python fetch_data.py

# Specify a stock
python fetch_data.py --symbol INFY --token 1594
```

Output:
```
Fetching 30 days of daily data for RELIANCE...

Date        Open      High      Low       Close     Volume
──────────────────────────────────────────────────────────────
2025-01-15  1285.00   1298.50   1278.30   1292.45   8234521
2025-01-16  1293.00   1305.75   1290.10   1301.20   9123456
...
```

### Run the analysis agent

```bash
# Default: RELIANCE
python agent.py

# Analyze a different stock
python agent.py --symbol TCS --token 11536
```

Output:
```
Connecting to AngelOne SmartAPI...
Fetching 30 days of data for RELIANCE (token: 2885)...
Sending data to Claude for analysis...

══════════════════════════════════════════════
  ANALYSIS REPORT: RELIANCE
══════════════════════════════════════════════

  Trend:          BULLISH
  Confidence:     High

  Support:        ₹1,278.30
  Resistance:     ₹1,342.80

  Avg Volume:     9,234,567
  Volume Trend:   Increasing

  Summary:
  RELIANCE has been in a steady uptrend over the past 30 days,
  gaining ~5.2%. The stock has formed higher lows consistently
  with strong volume support. Key support at ₹1,278 has held
  on multiple tests. Watch for breakout above ₹1,342 resistance.

══════════════════════════════════════════════
```

### Run the enhanced agent (with options data)

```bash
python agent_with_options.py --symbol RELIANCE --token 2885
```

This gives you everything above plus options market analysis — what IV and OI patterns suggest about market expectations.

---

## How to find symbol tokens

SmartAPI needs a "symbol token" (a numeric ID) for each stock. Some common ones:

| Stock | Token |
|---|---|
| RELIANCE | 2885 |
| TCS | 11536 |
| INFY | 1594 |
| HDFCBANK | 1333 |
| ICICIBANK | 4963 |
| SBIN | 3045 |
| BHARTIARTL | 10604 |
| ITC | 1660 |
| KOTAKBANK | 1922 |
| HINDUNILVR | 1394 |

For any other stock, check the [AngelOne instrument list](https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json) or search for it in your AngelOne trading app.

---

## What's next

This is intentionally simple. Here's where you could take it:

- **Option chain analysis** — `agent_with_options.py` is a start, but you could build a dedicated options screener that flags unusual OI buildup or IV skew
- **News sentiment** — Use Claude's web search tool or feed in news headlines to gauge sentiment alongside technicals
- **Automated alerts** — Run the agent on a cron job, send yourself Telegram/Slack alerts when it flags something interesting
- **Order placement** — `place_order.py` has the scaffolding. Once you're confident in your strategy, flip `DRY_RUN` to `False` (carefully)
- **Multi-agent setup** — Build specialized agents (a screener agent, a risk monitor, a news agent) and orchestrate them together
- **WebSocket streaming** — SmartAPI supports real-time tick data via WebSocket — useful for intraday strategies
- **Backtesting** — Pull historical data, run your agent's analysis backwards, see how its calls would have played out

---

## Disclaimer

This project is for **educational purposes only**. It is not financial advice.

- The analysis generated by Claude is based on historical price data and should not be used as the sole basis for trading decisions
- Always do your own research before making investment decisions
- The authors are not responsible for any financial losses incurred from using this code
- If you enable real order placement, you do so entirely at your own risk
- Past performance does not guarantee future results

---

## License

MIT — do whatever you want with it.

---

Built by [Anish](https://github.com/AM1403x). If you found this useful, star the repo.
