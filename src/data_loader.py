"""
Loads and validates Kaggle-style BTC-USD CSV datasets

Satisfies all project requirements:
  - File Upload     : accepts any .csv via Streamlit file_uploader
  - Auto Parsing    : detects Date OR Timestamp columns automatically
  - Price Selection : user picks Close / Open / High / Low from sidebar
  - Chronological   : sorts by date ascending after parsing
  - Missing days    : detects and forward-fills missing trading days
"""

import pandas as pd
from typing import Tuple, Optional


# All date/timestamp column names seen across common Kaggle BTC datasets
DATE_COLUMN_CANDIDATES = [
    "Date", "date",
    "Timestamp", "timestamp",
    "Datetime", "datetime",
    "Time", "time",
    "DATE", "TIMESTAMP",
]

# All price column names to offer in the sidebar
OHLC_COLUMNS = ["Close", "Open", "High", "Low", "Adj Close"]

# Date string formats to try (in order) if pandas cannot auto-detect
DATE_FORMATS = [
    "%Y-%m-%d",            # 2014-09-17         
    "%Y-%m-%d %H:%M:%S",   # 2014-09-17 00:00:00
    "%d/%m/%Y",            # 17/09/2014
    "%m/%d/%Y",            # 09/17/2014
    "%Y/%m/%d",            # 2014/09/17
    "%d-%m-%Y",            # 17-09-2014
    "%b %d, %Y",           # Sep 17, 2014
]



def load_and_validate_data( uploaded_file , price_col: str = "Close") -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Load and validate a Kaggle BTC CSV file
    Automatically detects the date column (Date or Timestamp)
    parses it to datetime64 and selects the user-chosen price column
    sorts chronologically and fills any missing trading days

    Parameters:
    uploaded_file : Streamlit UploadedFile
    price_col     : str — one of Close / Open / High / Low / Adj Close

    Returns:
    (df, None)       success — df columns: ds (datetime64), <price_col> (float64)
    (None, message)  failure — message shown directly via st.error() in app.py
    """

    # 1. Read CSV 
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        return None, f"Could not read the file: {e}"

    if df.empty:
        return None, "The uploaded file is empty"
    


    # 2. Detect date / timestamp column 
    date_col = _detect_date_column(df)
    if date_col is None:
        return None, (
            f"No date or timestamp column found in your file "
            f"Expected one of: {DATE_COLUMN_CANDIDATES} "
            f"Columns found: {list(df.columns)}"
        )



    # 3. Detect price column 
    actual_price_col = _detect_price_column(df, price_col)
    if actual_price_col is None:
        available = [c for c in OHLC_COLUMNS if c in df.columns]
        return None, (
            f"Price column '{price_col}' not found "
            f"Available price columns in your file: {available}"
        )



    # 4. Parse date column to datetime64 and rename to 'ds'
    df, date_error = _parse_date_column(df, date_col)
    if date_error:
        return None, date_error



    # 5. Parse price column to float64 
    df, price_error = _parse_price_column(df, actual_price_col)
    if price_error:
        return None, price_error



    # 6. Keep only the two working columns 
    # df = df[["ds", actual_price_col]].copy()

    cols_to_keep = ["ds", actual_price_col]
    for col in ["Open", "High", "Low", "Adj Close", "Volume"]:
        if col in df.columns and col != actual_price_col:
            cols_to_keep.append(col)
    df = df[cols_to_keep].copy()


    # 7. Sort chronologically 
    df = df.sort_values("ds").reset_index(drop = True)


    # 8. Minimum size check 
    if len(df) < 60:
        return None, (
            f"Dataset has only {len(df)} rows"
            "Need at least 60 daily rows to train a reliable model"
        )


    # 9. Detect and fill missing trading days 
    df, missing_days = _fill_missing_days(df, actual_price_col)


    # 10. Final sanity checks 
    df = df.dropna(subset = [actual_price_col])
    df = df[df[actual_price_col] > 0].reset_index(drop = True)

    if len(df) < 60:
        return None, (
            "Too many invalid rows after cleaning"
            "Please check your CSV for data quality issues"
        )

    return df, None




def get_available_price_columns(uploaded_file) -> list:
    """
    Returns the OHLC columns present in the uploaded file
    Reads only 2 rows fast regardless of file size
    Resets the file pointer so the file can be fully read afterward
    """
    try:
        df_head = pd.read_csv(uploaded_file, nrows=2)
        uploaded_file.seek(0)
        available = [c for c in OHLC_COLUMNS if c in df_head.columns]
        return available if available else ["Close", "Open", "High", "Low"]
    
    except Exception:
        uploaded_file.seek(0)
        return ["Close", "Open", "High", "Low"]


# Private helpers

def _detect_date_column(df: pd.DataFrame) -> Optional[str]:
    """
    Scans column names for any known date/timestamp variant
    Returns the first match or None if nothing found
    """
    for candidate in DATE_COLUMN_CANDIDATES:
        if candidate in df.columns:
            return candidate
    return None




def _detect_price_column(df: pd.DataFrame, requested: str) -> Optional[str]:
    """
    Returns the actual column name matching the requested price column
    Tries exact match first then case-insensitive match
    """
    if requested in df.columns:
        return requested
    
    lower_map = {c.lower(): c for c in df.columns}
    return lower_map.get(requested.lower())




def _parse_date_column( df: pd.DataFrame, date_col: str) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Parses the detected date column into datetime64 and renames it to 'ds'

    Handles:
      - Standard date strings  : "2014-09-17"  
      - Datetime strings       : "2014-09-17 00:00:00"
      - Alternative formats    : DD/MM/YYYY, MM/DD/YYYY
      - Unix timestamps (int)  : seconds or milliseconds
    """
    raw = df[date_col]

    # Unix timestamp (numeric column)
    if pd.api.types.is_numeric_dtype(raw):
        try:
            sample = raw.dropna().iloc[0]
            unit = "ms" if sample > 1e10 else "s"
            parsed = pd.to_datetime(raw, unit = unit, utc = True).dt.tz_localize(None)
            df = df.copy()
            df["ds"] = parsed
            return df, None
        except Exception as e:
            return df, f"Could not parse numeric timestamp column '{date_col}': {e}"


    # String date 
    try:
        parsed = pd.to_datetime(raw, infer_datetime_format = True)
        df = df.copy()
        df["ds"] = parsed
        return df, None
    except Exception:
        pass

    # String date — try explicit formats as fallback
    for fmt in DATE_FORMATS:
        try:
            parsed = pd.to_datetime(raw, format = fmt)
            df = df.copy()
            df["ds"] = parsed
            return df, None
        except Exception:
            continue

    sample_vals = raw.dropna().head(3).tolist()
    return df, (
        f"Could not parse date column '{date_col}'"
        f"Sample values: {sample_vals} "
        "Ensure the date column is in a standard format (e.g. YYYY-MM-DD)"
    )




def _parse_price_column( df: pd.DataFrame, price_col: str) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Coerces the price column to float64
    but handles edge cases like comma-formatted numbers ("45,000.50") and currency symbols
    """
    if pd.api.types.is_numeric_dtype(df[price_col]):
        df[price_col] = df[price_col].astype(float)
        return df, None

    try:
        df[price_col] = (
            df[price_col]
            .astype(str)
            .str.replace(",", "", regex = False)
            .str.replace("$", "", regex = False)
            .str.strip()
            .pipe(pd.to_numeric, errors="coerce")
        )
        return df, None
    except Exception as e:
        return df, f"Could not convert '{price_col}' to numeric: {e}"


def _fill_missing_days(df: pd.DataFrame, price_col: str) -> Tuple[pd.DataFrame, int]:
    """
    Detects and fills missing trading days by reindexing to a
    complete daily date range and forward-filling price values
    Returns the filled DataFrame and the count of days that were filled
    which can be shown as an info message in the UI
    """

    df = df.set_index("ds")
    full_range = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq="D",
    )

    missing_days = len(full_range) - len(df)

    df = df.reindex(full_range)

    df[price_col] = (
    df[price_col]
    .ffill()
    .bfill()
)

    df = df.reset_index().rename(columns = {"index": "ds"})

    return df, missing_days