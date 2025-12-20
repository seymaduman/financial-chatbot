"""
Financial Statements Scraper - Scrapes balance sheets, income statements, and cash flow
Extracts key metrics for financial analysis
"""
import re
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field
import os

import requests
from bs4 import BeautifulSoup

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import get_config


@dataclass
class FinancialMetrics:
    """Key financial metrics"""
    revenue: Optional[float] = None
    revenue_growth: Optional[float] = None
    net_income: Optional[float] = None
    gross_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    profit_margin: Optional[float] = None
    total_assets: Optional[float] = None
    total_liabilities: Optional[float] = None
    total_equity: Optional[float] = None
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    free_cash_flow: Optional[float] = None
    operating_cash_flow: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class FinancialStatement:
    """Financial statement data structure"""
    symbol: str
    statement_type: str  # income, balance, cashflow
    period: str  # annual, quarterly
    fiscal_year: str
    data: Dict[str, float]
    currency: str = "USD"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def to_text(self) -> str:
        """Convert to text for embedding"""
        lines = [f"\n{self.statement_type.upper()} STATEMENT - {self.symbol}"]
        lines.append(f"Period: {self.period.upper()} | Fiscal Year: {self.fiscal_year}")
        lines.append(f"Currency: {self.currency}")
        lines.append("-" * 40)
        
        for key, value in self.data.items():
            if value is not None:
                formatted = self._format_value(value)
                lines.append(f"{key}: {formatted}")
        
        return "\n".join(lines)
    
    def _format_value(self, value: float) -> str:
        """Format large numbers"""
        if abs(value) >= 1e12:
            return f"${value/1e12:.2f}T"
        elif abs(value) >= 1e9:
            return f"${value/1e9:.2f}B"
        elif abs(value) >= 1e6:
            return f"${value/1e6:.2f}M"
        elif abs(value) >= 1e3:
            return f"${value/1e3:.2f}K"
        else:
            return f"${value:.2f}"


class StatementsScraper:
    """
    Scraper for financial statements from Yahoo Finance
    Collects income statements, balance sheets, and cash flow statements
    """
    
    BASE_URL = "https://finance.yahoo.com"
    
    STATEMENT_URLS = {
        "income": "/quote/{symbol}/financials",
        "balance": "/quote/{symbol}/balance-sheet",
        "cashflow": "/quote/{symbol}/cash-flow"
    }
    
    def __init__(self):
        self.config = get_config()
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": self.config.scraping.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        })
        self._cache: Dict[str, Any] = {}
    
    def get_income_statement(self, symbol: str, annual: bool = True) -> Optional[FinancialStatement]:
        """
        Get income statement for a company
        
        Args:
            symbol: Stock ticker symbol
            annual: True for annual, False for quarterly
            
        Returns:
            FinancialStatement object or None
        """
        return self._get_statement(symbol, "income", annual)
    
    def get_balance_sheet(self, symbol: str, annual: bool = True) -> Optional[FinancialStatement]:
        """
        Get balance sheet for a company
        
        Args:
            symbol: Stock ticker symbol
            annual: True for annual, False for quarterly
            
        Returns:
            FinancialStatement object or None
        """
        return self._get_statement(symbol, "balance", annual)
    
    def get_cash_flow(self, symbol: str, annual: bool = True) -> Optional[FinancialStatement]:
        """
        Get cash flow statement for a company
        
        Args:
            symbol: Stock ticker symbol
            annual: True for annual, False for quarterly
            
        Returns:
            FinancialStatement object or None
        """
        return self._get_statement(symbol, "cashflow", annual)
    
    def _get_statement(self, symbol: str, statement_type: str, annual: bool) -> Optional[FinancialStatement]:
        """Get a specific financial statement"""
        cache_key = f"{symbol}_{statement_type}_{'annual' if annual else 'quarterly'}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            url_path = self.STATEMENT_URLS.get(statement_type)
            if not url_path:
                return None
            
            url = f"{self.BASE_URL}{url_path.format(symbol=symbol)}"
            if not annual:
                url += "?p=" + symbol + "&guccounter=1"
            
            response = self.session.get(url, timeout=self.config.scraping.request_timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, "lxml")
            statement = self._parse_statement_page(symbol, statement_type, annual, soup)
            
            if statement:
                self._cache[cache_key] = statement
            
            return statement
            
        except Exception as e:
            print(f"Error fetching {statement_type} statement for {symbol}: {e}")
            return None
    
    def _parse_statement_page(
        self, 
        symbol: str, 
        statement_type: str, 
        annual: bool, 
        soup: BeautifulSoup
    ) -> Optional[FinancialStatement]:
        """Parse financial statement from HTML"""
        data = {}
        fiscal_year = ""
        
        # Try to find data in embedded JSON first
        scripts = soup.find_all("script")
        for script in scripts:
            if script.string and "root.App.main" in script.string:
                try:
                    match = re.search(r'root\.App\.main\s*=\s*({.*?});', script.string, re.DOTALL)
                    if match:
                        json_data = json.loads(match.group(1))
                        result = self._extract_statement_from_json(json_data, statement_type, annual)
                        if result:
                            data, fiscal_year = result
                            break
                except:
                    continue
        
        # Fallback to HTML parsing if JSON extraction failed
        if not data:
            data, fiscal_year = self._parse_statement_html(soup, statement_type)
        
        if not data:
            return None
        
        return FinancialStatement(
            symbol=symbol,
            statement_type=statement_type,
            period="annual" if annual else "quarterly",
            fiscal_year=fiscal_year or datetime.now().strftime("%Y"),
            data=data
        )
    
    def _extract_statement_from_json(
        self, 
        json_data: dict, 
        statement_type: str, 
        annual: bool
    ) -> Optional[tuple]:
        """Extract financial data from embedded JSON"""
        try:
            stores = json_data.get("context", {}).get("dispatcher", {}).get("stores", {})
            
            # Map statement type to store key
            store_keys = {
                "income": "incomeStatementHistory" if annual else "incomeStatementHistoryQuarterly",
                "balance": "balanceSheetHistory" if annual else "balanceSheetHistoryQuarterly",
                "cashflow": "cashflowStatementHistory" if annual else "cashflowStatementHistoryQuarterly"
            }
            
            quote_summary = stores.get("QuoteSummaryStore", {})
            store_key = store_keys.get(statement_type)
            
            if not store_key:
                return None
            
            statement_data = quote_summary.get(store_key, {}).get(store_key.replace("History", "Statements"), [])
            
            if not statement_data:
                # Try alternative structure
                parent_key = store_key.replace("History", "") + "History"
                statement_data = quote_summary.get(parent_key, {}).get(store_key.replace("History", "Statements"), [])
            
            if not statement_data:
                return None
            
            # Get most recent statement
            latest = statement_data[0] if statement_data else {}
            
            # Extract fiscal year
            fiscal_year = ""
            if "endDate" in latest:
                end_date = latest["endDate"]
                if isinstance(end_date, dict):
                    fiscal_year = end_date.get("fmt", "")
                else:
                    fiscal_year = str(end_date)
            
            # Extract all available metrics
            data = {}
            for key, value in latest.items():
                if isinstance(value, dict) and "raw" in value:
                    # Convert camelCase to readable format
                    readable_key = self._camel_to_readable(key)
                    data[readable_key] = value["raw"]
            
            return data, fiscal_year
            
        except Exception as e:
            print(f"Error extracting statement from JSON: {e}")
            return None
    
    def _parse_statement_html(self, soup: BeautifulSoup, statement_type: str) -> tuple:
        """Parse financial statement from HTML tables"""
        data = {}
        fiscal_year = ""
        
        try:
            # Find the financial data table
            tables = soup.find_all("table")
            
            for table in tables:
                rows = table.find_all("tr")
                if len(rows) < 2:
                    continue
                
                # Get column headers (dates)
                header_row = rows[0]
                headers = [th.text.strip() for th in header_row.find_all(["th", "td"])]
                
                if len(headers) > 1:
                    fiscal_year = headers[1]  # Most recent period
                
                # Parse each row
                for row in rows[1:]:
                    cells = row.find_all(["th", "td"])
                    if len(cells) >= 2:
                        label = cells[0].text.strip()
                        value = cells[1].text.strip()
                        
                        if label and value:
                            parsed_value = self._parse_financial_value(value)
                            if parsed_value is not None:
                                data[label] = parsed_value
                
                if data:
                    break
            
        except Exception as e:
            print(f"Error parsing HTML: {e}")
        
        return data, fiscal_year
    
    def _parse_financial_value(self, text: str) -> Optional[float]:
        """Parse financial value from text"""
        try:
            # Clean the text
            text = text.replace(",", "").replace("$", "").strip()
            
            if text in ["--", "N/A", ""]:
                return None
            
            # Handle parentheses for negative values
            if text.startswith("(") and text.endswith(")"):
                text = "-" + text[1:-1]
            
            # Handle suffixes
            multiplier = 1
            if text.endswith("B"):
                multiplier = 1e9
                text = text[:-1]
            elif text.endswith("M"):
                multiplier = 1e6
                text = text[:-1]
            elif text.endswith("K"):
                multiplier = 1e3
                text = text[:-1]
            elif text.endswith("T"):
                multiplier = 1e12
                text = text[:-1]
            
            return float(text) * multiplier
            
        except:
            return None
    
    def _camel_to_readable(self, text: str) -> str:
        """Convert camelCase to readable format"""
        # Insert space before uppercase letters
        result = re.sub(r'([A-Z])', r' \1', text)
        # Capitalize first letter
        return result.strip().title()
    
    def get_key_metrics(self, symbol: str) -> FinancialMetrics:
        """
        Calculate key financial metrics from statements
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            FinancialMetrics object with calculated ratios
        """
        income = self.get_income_statement(symbol)
        balance = self.get_balance_sheet(symbol)
        cashflow = self.get_cash_flow(symbol)
        
        metrics = FinancialMetrics()
        
        # Extract from income statement
        if income and income.data:
            metrics.revenue = income.data.get("Total Revenue")
            metrics.net_income = income.data.get("Net Income") or income.data.get("Net Income Common Stockholders")
            
            gross_profit = income.data.get("Gross Profit")
            if gross_profit and metrics.revenue:
                metrics.gross_margin = (gross_profit / metrics.revenue) * 100
            
            operating_income = income.data.get("Operating Income")
            if operating_income and metrics.revenue:
                metrics.operating_margin = (operating_income / metrics.revenue) * 100
            
            if metrics.net_income and metrics.revenue:
                metrics.profit_margin = (metrics.net_income / metrics.revenue) * 100
        
        # Extract from balance sheet
        if balance and balance.data:
            metrics.total_assets = balance.data.get("Total Assets")
            metrics.total_liabilities = balance.data.get("Total Liabilities Net Minority Interest") or balance.data.get("Total Liabilities")
            metrics.total_equity = balance.data.get("Total Equity Gross Minority Interest") or balance.data.get("Total Stockholders Equity")
            
            total_debt = balance.data.get("Total Debt")
            if total_debt and metrics.total_equity and metrics.total_equity > 0:
                metrics.debt_to_equity = (total_debt / metrics.total_equity) * 100
            
            current_assets = balance.data.get("Current Assets")
            current_liabilities = balance.data.get("Current Liabilities")
            if current_assets and current_liabilities and current_liabilities > 0:
                metrics.current_ratio = current_assets / current_liabilities
        
        # Extract from cash flow
        if cashflow and cashflow.data:
            metrics.operating_cash_flow = cashflow.data.get("Operating Cash Flow") or cashflow.data.get("Cash Flow From Operating Activities")
            metrics.free_cash_flow = cashflow.data.get("Free Cash Flow")
            
            if not metrics.free_cash_flow:
                capex = cashflow.data.get("Capital Expenditure") or cashflow.data.get("Capital Expenditures")
                if metrics.operating_cash_flow and capex:
                    metrics.free_cash_flow = metrics.operating_cash_flow - abs(capex)
        
        return metrics
    
    def get_all_statements(self, symbol: str) -> Dict[str, Optional[FinancialStatement]]:
        """
        Get all financial statements for a company
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dictionary with all statements
        """
        return {
            "income_annual": self.get_income_statement(symbol, annual=True),
            "income_quarterly": self.get_income_statement(symbol, annual=False),
            "balance_annual": self.get_balance_sheet(symbol, annual=True),
            "balance_quarterly": self.get_balance_sheet(symbol, annual=False),
            "cashflow_annual": self.get_cash_flow(symbol, annual=True),
            "cashflow_quarterly": self.get_cash_flow(symbol, annual=False)
        }
    
    def get_financial_summary(self, symbol: str) -> str:
        """
        Get a text summary of financial health
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Text summary suitable for RAG
        """
        metrics = self.get_key_metrics(symbol)
        income = self.get_income_statement(symbol)
        balance = self.get_balance_sheet(symbol)
        
        lines = [f"\n=== FINANCIAL SUMMARY: {symbol} ===\n"]
        
        # Revenue and profitability
        lines.append("PROFITABILITY:")
        if metrics.revenue:
            lines.append(f"  Revenue: {self._format_number(metrics.revenue)}")
        if metrics.net_income:
            lines.append(f"  Net Income: {self._format_number(metrics.net_income)}")
        if metrics.gross_margin:
            lines.append(f"  Gross Margin: {metrics.gross_margin:.1f}%")
        if metrics.operating_margin:
            lines.append(f"  Operating Margin: {metrics.operating_margin:.1f}%")
        if metrics.profit_margin:
            lines.append(f"  Profit Margin: {metrics.profit_margin:.1f}%")
        
        # Balance sheet health
        lines.append("\nBALANCE SHEET:")
        if metrics.total_assets:
            lines.append(f"  Total Assets: {self._format_number(metrics.total_assets)}")
        if metrics.total_liabilities:
            lines.append(f"  Total Liabilities: {self._format_number(metrics.total_liabilities)}")
        if metrics.total_equity:
            lines.append(f"  Total Equity: {self._format_number(metrics.total_equity)}")
        if metrics.debt_to_equity:
            lines.append(f"  Debt-to-Equity: {metrics.debt_to_equity:.1f}%")
        if metrics.current_ratio:
            lines.append(f"  Current Ratio: {metrics.current_ratio:.2f}")
        
        # Cash flow
        lines.append("\nCASH FLOW:")
        if metrics.operating_cash_flow:
            lines.append(f"  Operating Cash Flow: {self._format_number(metrics.operating_cash_flow)}")
        if metrics.free_cash_flow:
            lines.append(f"  Free Cash Flow: {self._format_number(metrics.free_cash_flow)}")
        
        # Health assessment
        lines.append("\nHEALTH ASSESSMENT:")
        health_score = self._assess_financial_health(metrics)
        lines.append(f"  Overall Health: {health_score}")
        
        return "\n".join(lines)
    
    def _format_number(self, value: float) -> str:
        """Format large numbers"""
        if value is None:
            return "N/A"
        if abs(value) >= 1e12:
            return f"${value/1e12:.2f}T"
        elif abs(value) >= 1e9:
            return f"${value/1e9:.2f}B"
        elif abs(value) >= 1e6:
            return f"${value/1e6:.2f}M"
        else:
            return f"${value:,.0f}"
    
    def _assess_financial_health(self, metrics: FinancialMetrics) -> str:
        """Provide a simple health assessment"""
        score = 0
        factors = []
        
        # Profitability
        if metrics.profit_margin:
            if metrics.profit_margin > 15:
                score += 2
                factors.append("Strong profit margin")
            elif metrics.profit_margin > 5:
                score += 1
                factors.append("Moderate profit margin")
            else:
                factors.append("Low profit margin")
        
        # Debt
        if metrics.debt_to_equity:
            if metrics.debt_to_equity < 50:
                score += 2
                factors.append("Low debt")
            elif metrics.debt_to_equity < 100:
                score += 1
                factors.append("Moderate debt")
            else:
                factors.append("High debt levels")
        
        # Liquidity
        if metrics.current_ratio:
            if metrics.current_ratio > 2:
                score += 2
                factors.append("Strong liquidity")
            elif metrics.current_ratio > 1:
                score += 1
                factors.append("Adequate liquidity")
            else:
                factors.append("Liquidity concerns")
        
        # Cash flow
        if metrics.free_cash_flow and metrics.free_cash_flow > 0:
            score += 2
            factors.append("Positive free cash flow")
        elif metrics.free_cash_flow and metrics.free_cash_flow < 0:
            factors.append("Negative free cash flow")
        
        # Determine overall health
        if score >= 6:
            health = "STRONG ✓"
        elif score >= 4:
            health = "MODERATE"
        else:
            health = "CONCERNING ⚠"
        
        return f"{health} ({', '.join(factors[:3])})"


if __name__ == "__main__":
    # Test the statements scraper
    scraper = StatementsScraper()
    
    print("Testing Financial Statements Scraper...")
    print("-" * 50)
    
    symbol = "AAPL"
    
    # Test income statement
    print(f"\nIncome Statement for {symbol}:")
    income = scraper.get_income_statement(symbol)
    if income:
        print(income.to_text()[:500])
    else:
        print("  Failed to retrieve")
    
    # Test balance sheet
    print(f"\nBalance Sheet for {symbol}:")
    balance = scraper.get_balance_sheet(symbol)
    if balance:
        print(balance.to_text()[:500])
    else:
        print("  Failed to retrieve")
    
    # Test key metrics
    print(f"\nKey Metrics for {symbol}:")
    metrics = scraper.get_key_metrics(symbol)
    print(f"  Revenue: {scraper._format_number(metrics.revenue)}")
    print(f"  Profit Margin: {metrics.profit_margin:.1f}%" if metrics.profit_margin else "  Profit Margin: N/A")
    print(f"  Debt/Equity: {metrics.debt_to_equity:.1f}%" if metrics.debt_to_equity else "  Debt/Equity: N/A")
    
    # Test financial summary
    print("\n" + "-" * 50)
    print(scraper.get_financial_summary(symbol))
