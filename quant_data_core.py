# quant_data_core.py - ENHANCED VERSION
import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import quant_paths

SLOT_NAMES = {1: "FRBD", 2: "GZBD", 3: "GALI", 4: "DSWR"}
SLOT_IDS = {v: k for k, v in SLOT_NAMES.items()}

def _parse_cutoff_date(value):
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.to_datetime(value, errors="coerce").date()
    if isinstance(value, str):
        raw = value.strip()
        for fmt in ("%Y-%m-%d", "%d-%m-%y", "%d-%m-%Y"):
            try:
                return datetime.strptime(raw, fmt).date()
            except ValueError:
                continue
    return None


def load_results_dataframe(results_file: Path | str | None = None, verbose: bool = False, cutoff_date=None):
    """
    Load central results file in a robust way.

    - Handles files with or without a header row.
    - Forces columns: DATE, FRBD, GZBD, GALI, DSWR.
    - Parses DATE from Excel serials, timestamps, or strings.
    - Ensures slot columns are numeric.
    """
    results_file = Path(results_file) if results_file else quant_paths.get_results_file_path()

    try:
        df_raw = pd.read_excel(results_file, header=None)
        if verbose:
            print(f"Found columns: {df_raw.columns.tolist()}")
            print(f"Raw shape: {df_raw.shape}")
    except Exception as e:
        print(f"❌ Error loading real results: {e}")
        return pd.DataFrame()

    if df_raw.empty:
        print("❌ Real results file is empty")
        return pd.DataFrame()

    def _is_datetime_like(value):
        if pd.isna(value):
            return False
        if isinstance(value, (pd.Timestamp, datetime, np.datetime64)):
            return True
        # Excel serials or other numeric encodings
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return True
        if isinstance(value, str):
            try:
                parsed = pd.to_datetime(value, errors="raise")
                return not pd.isna(parsed)
            except Exception:
                return False
        return False

    # Decide if first row is header or data
    first_row = df_raw.iloc[0]
    first_cell = first_row.iloc[0]
    header_is_data = _is_datetime_like(first_cell)

    if header_is_data:
        if verbose:
            print("ℹ️  Detected first row as data (no header row present)")
        df = df_raw.iloc[:, :5].copy()
    else:
        if verbose:
            print("ℹ️  Detected header row; normalizing column names")
        inferred_columns = [str(col).strip().upper() for col in first_row]
        df = df_raw.iloc[1:, :5].copy()
        df.columns = inferred_columns

    # Force final columns regardless of path
    df = df.iloc[:, :5].copy()
    df.columns = ["DATE", "FRBD", "GZBD", "GALI", "DSWR"]

    # Robust DATE parsing
    excel_origin = datetime(1899, 12, 30).date()

    def _parse_date_value(value):
        if pd.isna(value):
            return None

        # Already datetime-like
        if isinstance(value, (pd.Timestamp, datetime, np.datetime64)):
            try:
                return pd.to_datetime(value, errors="coerce").date()
            except Exception:
                return None

        # Excel serial / numeric
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            try:
                return excel_origin + timedelta(days=float(value))
            except Exception:
                return None

        # String
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return None
            try:
                return pd.to_datetime(value, errors="raise").date()
            except Exception:
                return None

        return None

    parsed_dates = []
    invalid_date_values = []

    for raw_val in df["DATE"].tolist():
        parsed = _parse_date_value(raw_val)
        parsed_dates.append(parsed)
        if parsed is None:
            invalid_date_values.append(raw_val)

    if invalid_date_values:
        sample_values = invalid_date_values[:5]
        print(f"⚠️  Failed to parse {len(invalid_date_values)} DATE entries. Samples: {sample_values}")

    df["DATE"] = pd.to_datetime(parsed_dates, errors="coerce")
    invalid_after_parse = df["DATE"].isna().sum()

    if invalid_after_parse:
        print(f"⚠️  Dropping {invalid_after_parse} rows with unparseable DATE values")
        df = df.dropna(subset=["DATE"])

    # If still nothing valid, bail out gracefully
    if df.empty:
        print("❌ No valid DATE values found after parsing; exiting gracefully")
        return pd.DataFrame()

    # Drop obviously bogus historical dates (e.g. Excel serial 0 -> 1899-12-31)
    min_valid_date = datetime(2000, 1, 1)
    bogus_mask = df["DATE"] < pd.Timestamp(min_valid_date)
    if bogus_mask.any():
        dropped_bogus = int(bogus_mask.sum())
        print(f"⚠️  Dropping {dropped_bogus} rows with implausible DATE < {min_valid_date.date()}")
        df = df.loc[~bogus_mask].copy()

    if df.empty:
        print("❌ No valid DATE values left after DATE sanity filter; exiting gracefully")
        return pd.DataFrame()

    # Ensure slot columns exist and are numeric
    for slot in ["FRBD", "GZBD", "GALI", "DSWR"]:
        if slot not in df.columns:
            print(f"⚠️  Slot column '{slot}' not found, creating with NaN values")
            df[slot] = np.nan
        df.loc[:, slot] = pd.to_numeric(df[slot], errors="coerce")

    # Final sanity + logging
    total_rows = len(df)
    if not pd.api.types.is_datetime64_any_dtype(df["DATE"]):
        # Even if dtype is 'object', as long as values are datetime-like, we can still work.
        df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")

    unique_dates = df["DATE"].dt.date.dropna().unique()
    if len(unique_dates) == 0:
        print("❌ No valid DATE values found after final DATE conversion")
        return pd.DataFrame()

    cutoff_value = cutoff_date
    if cutoff_value is None:
        cutoff_value = os.environ.get("PREDICTOR_CUTOFF_DATE")
    cutoff = _parse_cutoff_date(cutoff_value)
    if cutoff is not None:
        df = df[df["DATE"].dt.date <= cutoff]

    df = df.sort_values("DATE").reset_index(drop=True)

    if verbose:
        print(f"✅ Loaded results data: {total_rows} records with columns: {df.columns.tolist()}")
        print(f"📅 DATE range: {df['DATE'].min().date()} to {df['DATE'].max().date()}")

    return df

def get_latest_result_date(df):
    """Get the latest result date from DataFrame - ROBUST VERSION"""
    if df.empty or 'DATE' not in df.columns:
        return None
    
    # Make a copy and ensure DATE is datetime
    df_temp = df.copy()
    df_temp['DATE'] = pd.to_datetime(df_temp['DATE'], errors='coerce')
    df_temp = df_temp.dropna(subset=['DATE'])
    
    if df_temp.empty:
        return None
    
    # Convert to date only
    df_temp['DATE_ONLY'] = df_temp['DATE'].dt.date
    today = datetime.now().date()
    
    # Filter to past or today dates
    past_or_today = df_temp[df_temp['DATE_ONLY'] <= today]
    
    if not past_or_today.empty:
        return past_or_today['DATE_ONLY'].max()
    
    # Fallback: if all dates are in future, use the most recent one anyway
    return df_temp['DATE_ONLY'].max()

def get_slot_fill_status_for_date(df, date):
    """Check which slots are filled for a given date"""
    if df.empty or 'DATE' not in df.columns:
        return {}
    
    date_data = df[df['DATE'] == date]
    if date_data.empty:
        return {}
    
    slot_status = {}
    slots = ['FRBD', 'GZBD', 'GALI', 'DSWR']
    
    for slot in slots:
        if slot in date_data.columns:
            value = date_data[slot].iloc[0]
            # Check if slot has valid result (not NaN and not empty string)
            slot_status[slot] = pd.notna(value) and value != ''
        else:
            slot_status[slot] = False
    
    return slot_status

def build_prediction_plan(df):
    """Build prediction plan based on latest results - ROBUST VERSION"""
    latest_date = get_latest_result_date(df)
    today = datetime.now().date()
    
    # Fallback if no valid date found
    if not latest_date:
        latest_date = today
        print("⚠️  No valid result data found, using today as fallback")
    
    slot_status = get_slot_fill_status_for_date(df, latest_date)
    
    # Determine if we're dealing with today or a past day
    if latest_date == today:
        # Today - check which slots are filled
        filled_slots = [slot for slot, filled in slot_status.items() if filled]
        all_slots = ['FRBD', 'GZBD', 'GALI', 'DSWR']
        remaining_slots = [slot for slot in all_slots if slot not in filled_slots]
        
        if remaining_slots:
            # Partial day - predict remaining slots for today
            is_partial_day = True
            today_slots_to_predict = remaining_slots
            next_date = today
            mode = "partial_today"
        else:
            # Full day completed - predict next day
            is_partial_day = False
            today_slots_to_predict = []
            next_date = today + timedelta(days=1)
            mode = "next_day"
    else:
        # Past day - predict next day
        is_partial_day = False
        today_slots_to_predict = []
        next_date = latest_date + timedelta(days=1)
        mode = "next_day"
    
    plan = {
        'latest_result_date': latest_date,
        'is_partial_day': is_partial_day,
        'today_slots_to_predict': today_slots_to_predict,
        'next_date': next_date,
        'mode': mode,
        'slot_status': slot_status
    }
    
    return plan

def print_prediction_plan_summary(plan):
    """Print human-readable prediction plan summary"""
    print("\n🎯 PREDICTION PLAN SUMMARY")
    print("=" * 40)
    print(f"Latest result date: {plan['latest_result_date']}")
    print(f"Plan mode: {plan['mode']}")
    
    if plan['is_partial_day']:
        print(f"Same-day slots to predict: {', '.join(plan['today_slots_to_predict'])}")
    else:
        print("Same-day slots: None")
        
    print(f"Next prediction date: {plan['next_date']}")
    
    if plan['slot_status']:
        print("Slot status:")
        for slot, filled in plan['slot_status'].items():
            status = "✅ Filled" if filled else "❌ Missing"
            print(f"  {slot}: {status}")

# Utility function for date handling
def get_date_range(days_back=30):
    """Get date range for analysis"""
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days_back)
    return start_date, end_date

def filter_data_by_date(df, start_date, end_date):
    """Filter DataFrame by date range"""
    if 'DATE' not in df.columns:
        return df

    mask = (df['DATE'] >= start_date) & (df['DATE'] <= end_date)
    return df[mask]


def _results_to_long(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize results to a long format with numeric slots.

    This helper accepts either wide (DATE + FRBD/GZBD/GALI/DSWR) or long
    (date/slot/number) dataframes and produces a consistent structure.
    """

    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    if "date" not in df.columns and "DATE" in df.columns:
        df["date"] = pd.to_datetime(df["DATE"], errors="coerce")

    # Already long
    if "slot" in df.columns and "number" in df.columns:
        df["slot"] = df["slot"].apply(lambda x: SLOT_IDS.get(x, SLOT_IDS.get(str(x).upper(), x)))
        df["number"] = pd.to_numeric(df["number"], errors="coerce")
        df = df.dropna(subset=["date", "slot", "number"])
        df["slot"] = df["slot"].astype(int)
        df["number"] = df["number"].astype(int) % 100
        return df[["date", "slot", "number"]]

    slot_cols = [c for c in SLOT_IDS.keys() if c in df.columns]
    parts = []
    for col in slot_cols:
        part = df[["date", col]].copy()
        part = part.rename(columns={col: "number"})
        part["slot"] = SLOT_IDS[col]
        parts.append(part)

    if not parts:
        return pd.DataFrame(columns=["date", "slot", "number"])

    long_df = pd.concat(parts, ignore_index=True)
    long_df["number"] = pd.to_numeric(long_df["number"], errors="coerce")
    long_df = long_df.dropna(subset=["date", "slot", "number"])
    long_df["slot"] = long_df["slot"].astype(int)
    long_df["number"] = long_df["number"].astype(int) % 100
    return long_df[["date", "slot", "number"]]


def compute_learning_signals(df: pd.DataFrame, target_date=None, lookback_days: int = 7):
    """Aggregate learning signals for each slot/day.

    Signals include:
    - Recent exact hits for the same slot (strong boost).
    - Cross-slot hits from the previous day.
    - Mirror and neighbor effects from the previous day.
    - Soft weekly memory for any appearance in the lookback window.
    """

    df_long = _results_to_long(df)
    if df_long.empty:
        return {slot: {"multipliers": {}, "details": {}} for slot in SLOT_NAMES.values()}

    if target_date is None:
        target_date = (df_long["date"].max() + pd.Timedelta(days=1)).date()

    history = df_long[df_long["date"].dt.date < target_date].copy()
    if history.empty:
        return {slot: {"multipliers": {}, "details": {}} for slot in SLOT_NAMES.values()}

    start_window = target_date - timedelta(days=lookback_days)
    recent_window = history[history["date"].dt.date >= start_window]
    prev_day = target_date - timedelta(days=1)
    prev_day_rows = history[history["date"].dt.date == prev_day]

    signals = {
        slot: {"multipliers": {}, "details": {}} for slot in SLOT_NAMES.values()
    }

    def _bump(slot_name: str, num: int, weight: float, tag: str):
        multipliers = signals[slot_name]["multipliers"]
        details = signals[slot_name]["details"]
        current = multipliers.get(num, 1.0)
        multipliers[num] = current * weight
        if tag not in details.get(num, []):
            details.setdefault(num, []).append(tag)

    # Strong signals from the previous day
    for _, row in prev_day_rows.iterrows():
        num = int(row["number"]) % 100
        slot_name = SLOT_NAMES.get(int(row["slot"]), str(row["slot"]))
        _bump(slot_name, num, 1.15, "recent_hit")

        for other_slot in SLOT_NAMES.values():
            if other_slot != slot_name:
                _bump(other_slot, num, 1.05, "cross_slot_prev_day")

        mirror = int(f"{num % 10}{num // 10}")
        _bump(slot_name, mirror, 1.05, "mirror_prev_day")
        _bump(slot_name, (num + 1) % 100, 1.03, "neighbor_prev_day")
        _bump(slot_name, (num - 1) % 100, 1.03, "neighbor_prev_day")

    # Softer memory across the weekly lookback
    for _, row in recent_window.iterrows():
        num = int(row["number"]) % 100
        slot_name = SLOT_NAMES.get(int(row["slot"]), str(row["slot"]))
        _bump(slot_name, num, 1.03, "weekly_memory")

    return signals


def apply_signal_multipliers(score_map: dict, slot_signal: dict):
    """Apply learning signal multipliers to a raw score map."""

    multipliers = slot_signal.get("multipliers", {}) if slot_signal else {}
    details = slot_signal.get("details", {}) if slot_signal else {}
    adjusted = {}
    notes = {}

    for num, score in score_map.items():
        multiplier = multipliers.get(num, 1.0)
        adjusted[num] = score * multiplier
        if multiplier != 1.0 and num in details:
            notes[num] = details[num]

    return adjusted, notes


def apply_learning_to_dataframe(
    pred_df: pd.DataFrame,
    signals: dict,
    *,
    slot_col: str = "slot",
    number_col: str = "number",
    rank_col: str | None = "rank",
    score_candidates: tuple = ("score", "confidence", "probability"),
    note_col: str = "learning_signals",
):
    """Re-rank and annotate predictions using learning signals."""

    if pred_df is None or pred_df.empty:
        return pred_df

    df = pred_df.copy()
    score_col = next((c for c in score_candidates if c in df.columns), None)

    def _slot_name(val):
        if pd.isna(val):
            return None
        key = str(val).upper()
        if key in SLOT_IDS:
            return key
        try:
            as_int = int(float(val))
            return SLOT_NAMES.get(as_int)
        except Exception:
            return key

    def _num_int(val):
        try:
            return int(str(val).zfill(2)) % 100
        except Exception:
            return None

    df["_slot_name"] = df[slot_col].apply(_slot_name)
    df["_number_int"] = df[number_col].apply(_num_int)

    if score_col:
        df[score_col] = pd.to_numeric(df[score_col], errors="coerce").fillna(0.0)
        df["_base_score"] = df[score_col]
        score_target = score_col
    else:
        if rank_col and rank_col in df.columns:
            max_rank = df[rank_col].max()
            df["_base_score"] = df[rank_col].apply(lambda r: (max_rank - int(r) + 1) if pd.notna(r) else 1.0)
        else:
            df["_base_score"] = 1.0
        score_target = score_candidates[0]

    df[note_col] = ""
    adjusted_scores = []
    for idx, row in df.iterrows():
        slot_name = row["_slot_name"]
        num = row["_number_int"]
        multiplier = signals.get(slot_name, {}).get("multipliers", {}).get(num, 1.0)
        note_list = signals.get(slot_name, {}).get("details", {}).get(num, [])
        adjusted_scores.append(row["_base_score"] * multiplier)
        if note_list:
            df.at[idx, note_col] = "; ".join(sorted(set(note_list)))

    df[score_target] = adjusted_scores

    sort_cols = [slot_col, score_target]
    ascending = [True, False]
    if rank_col and rank_col in df.columns:
        sort_cols.append(rank_col)
        ascending.append(True)

    df = df.sort_values(by=sort_cols, ascending=ascending).reset_index(drop=True)
    if rank_col and rank_col in df.columns:
        df[rank_col] = df.groupby(slot_col).cumcount() + 1

    df = df.drop(columns=["_slot_name", "_number_int", "_base_score"], errors="ignore")
    return df
