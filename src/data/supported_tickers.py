"""
Supported Tickers List
Contains top stocks from S&P 500, NASDAQ 100, and major Cryptocurrencies.
"""

SUPPORTED_TICKERS = [
    # Technology (Mag 7 + Others)
    {"label": "Apple Inc. (AAPL)", "value": "AAPL"},
    {"label": "Microsoft Corp. (MSFT)", "value": "MSFT"},
    {"label": "NVIDIA Corp. (NVDA)", "value": "NVDA"},
    {"label": "Alphabet Inc. (GOOGL)", "value": "GOOGL"},
    {"label": "Amazon.com Inc. (AMZN)", "value": "AMZN"},
    {"label": "Meta Platforms (META)", "value": "META"},
    {"label": "Tesla Inc. (TSLA)", "value": "TSLA"},
    {"label": "Broadcom Inc. (AVGO)", "value": "AVGO"},
    {"label": "AMD (AMD)", "value": "AMD"},
    {"label": "Intel Corp. (INTC)", "value": "INTC"},
    {"label": "Netflix Inc. (NFLX)", "value": "NFLX"},
    {"label": "Salesforce (CRM)", "value": "CRM"},
    {"label": "Adobe Inc. (ADBE)", "value": "ADBE"},
    {"label": "Oracle Corp. (ORCL)", "value": "ORCL"},
    {"label": "Cisco Systems (CSCO)", "value": "CSCO"},
    {"label": "IBM (IBM)", "value": "IBM"},
    {"label": "Qualcomm (QCOM)", "value": "QCOM"},
    {"label": "Texas Instruments (TXN)", "value": "TXN"},
    {"label": "Uber Technologies (UBER)", "value": "UBER"},
    {"label": "Airbnb (ABNB)", "value": "ABNB"},
    {"label": "Palantir (PLTR)", "value": "PLTR"},
    {"label": "Snowflake (SNOW)", "value": "SNOW"},

    # Finance
    {"label": "JPMorgan Chase (JPM)", "value": "JPM"},
    {"label": "Bank of America (BAC)", "value": "BAC"},
    {"label": "Wells Fargo (WFC)", "value": "WFC"},
    {"label": "Visa Inc. (V)", "value": "V"},
    {"label": "Mastercard (MA)", "value": "MA"},
    {"label": "PayPal (PYPL)", "value": "PYPL"},
    {"label": "Goldman Sachs (GS)", "value": "GS"},
    {"label": "Morgan Stanley (MS)", "value": "MS"},
    {"label": "BlackRock (BLK)", "value": "BLK"},

    # Consumer & Retail
    {"label": "Walmart (WMT)", "value": "WMT"},
    {"label": "Costco (COST)", "value": "COST"},
    {"label": "Home Depot (HD)", "value": "HD"},
    {"label": "Coca-Cola (KO)", "value": "KO"},
    {"label": "PepsiCo (PEP)", "value": "PEP"},
    {"label": "McDonald's (MCD)", "value": "MCD"},
    {"label": "Starbucks (SBUX)", "value": "SBUX"},
    {"label": "Nike (NKE)", "value": "NKE"},
    {"label": "Procter & Gamble (PG)", "value": "PG"},

    # Automotive & Industrial
    {"label": "Ford Motor Co. (F)", "value": "F"},
    {"label": "General Motors (GM)", "value": "GM"},
    {"label": "Boeing (BA)", "value": "BA"},
    {"label": "General Electric (GE)", "value": "GE"},
    {"label": "Caterpillar (CAT)", "value": "CAT"},
    {"label": "Lockheed Martin (LMT)", "value": "LMT"},

    # Healthcare
    {"label": "Johnson & Johnson (JNJ)", "value": "JNJ"},
    {"label": "Pfizer (PFE)", "value": "PFE"},
    {"label": "Merck & Co. (MRK)", "value": "MRK"},
    {"label": "Eli Lilly (LLY)", "value": "LLY"},
    {"label": "AbbVie (ABBV)", "value": "ABBV"},
    {"label": "UnitedHealth Group (UNH)", "value": "UNH"},

    # Energy
    {"label": "Exxon Mobil (XOM)", "value": "XOM"},
    {"label": "Chevron (CVX)", "value": "CVX"},
    {"label": "Shell (SHEL)", "value": "SHEL"},
    {"label": "BP (BP)", "value": "BP"},

    # Crypto
    {"label": "Bitcoin (BTC-USD)", "value": "BTC-USD"},
    {"label": "Ethereum (ETH-USD)", "value": "ETH-USD"},
    {"label": "Solana (SOL-USD)", "value": "SOL-USD"},
    {"label": "Ripple (XRP-USD)", "value": "XRP-USD"},
    {"label": "Dogecoin (DOGE-USD)", "value": "DOGE-USD"},
    {"label": "Cardano (ADA-USD)", "value": "ADA-USD"},

    # Indices
    {"label": "S&P 500 (^GSPC)", "value": "^GSPC"},
    {"label": "Dow Jones (^DJI)", "value": "^DJI"},
    {"label": "NASDAQ Composite (^IXIC)", "value": "^IXIC"},
    {"label": "Russell 2000 (^RUT)", "value": "^RUT"},
    {"label": "VIX Volatility (^VIX)", "value": "^VIX"},
    
    # Other Popular
    {"label": "GameStop (GME)", "value": "GME"},
    {"label": "AMC Entertainment (AMC)", "value": "AMC"},
    {"label": "Coinbase (COIN)", "value": "COIN"},
    {"label": "MicroStrategy (MSTR)", "value": "MSTR"},
]

def get_ticker_label(ticker_value):
    """Get label for a ticker value"""
    for item in SUPPORTED_TICKERS:
        if item["value"] == ticker_value:
            return item["label"]
    return ticker_value
