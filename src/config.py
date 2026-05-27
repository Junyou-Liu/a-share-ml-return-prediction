from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FACTOR_FILE = PROJECT_ROOT / "factors_6stocks_weekly_20251107_013757.csv"
BENCHMARK_FILE = PROJECT_ROOT / "bm_399006_weekly.csv"

DATE_CANDIDATES = ("date", "trade_date", "week")
CODE_CANDIDATES = ("stock_code", "code", "ticker")
PRICE_CANDIDATES = ("close_hfq", "adj_close", "close", "price", "px_close")

FACTOR_COLUMNS = (
    "BM",
    "EP",
    "EV_EBITDA",
    "FCF_Yield",
    "ROE",
    "Gross_Profit",
    "Accruals",
    "Asset_Growth",
    "NBIS_W",
    "ATVR_5_20",
    "REV1",
    "MOM_2_20",
    "IND_RS_5",
    "VOL20",
    "AMIHUD20",
    "THEME_HEAT",
    "RSI",
    "MACD",
)

WEEKS_PER_YEAR = 52

