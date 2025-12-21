"""Centralized Excel loader used by the DeepSeek scripts.

This module delegates the raw parsing to quant_data_core.load_results_dataframe
and then converts the results into a normalized long format with columns:
    - date: pandas.Timestamp (date component)
    - slot: integer slot identifier (1-4)
    - number: integer prediction value (0-99)
"""

from pathlib import Path
import pandas as pd

from quant_data_core import load_results_dataframe


SLOT_MAP = {
    "FRBD": 1,
    "GZBD": 2,
    "GALI": 3,
    "DSWR": 4,
}


def _call_loader_with_optional_path(file_path: Path):
    """Call ``load_results_dataframe`` while tolerating legacy signatures.

    Some historical versions of ``load_results_dataframe`` didn't accept a
    file path argument. In that scenario, fall back to calling it without
    parameters so older environments keep working.
    """

    try:
        return load_results_dataframe(file_path)
    except TypeError as exc:
        # Gracefully handle legacy signature ``load_results_dataframe()``
        if "takes 0 positional arguments" in str(exc):
            print("ℹ️  load_results_dataframe does not accept a file path; using default path")
            return load_results_dataframe()
        raise


def load_results_excel(file_path="number prediction learn.xlsx"):
    """Load the Excel results file and return a long-format DataFrame.

    Args:
        file_path (str | Path): Path to the Excel file. Defaults to the
            standard file name used by the DeepSeek scripts.

    Returns:
        pd.DataFrame: Columns [date, slot, number], sorted by date then slot.
    """
    df_raw = _call_loader_with_optional_path(Path(file_path))
    if df_raw.empty:
        print("❌ No data found in results file")
        return pd.DataFrame(columns=["date", "slot", "number"])

    # Normalize column names so we can flexibly accept both wide and already
    # melted data. This also trims stray spaces that occasionally appear in
    # Excel headers.
    df_raw = df_raw.rename(columns={col: str(col).strip().upper() for col in df_raw.columns})

    # Case 1: incoming data is already in long format (DATE/SLOT/NUMBER).
    if {"DATE", "SLOT", "NUMBER"}.issubset(df_raw.columns):
        long_df = df_raw[["DATE", "SLOT", "NUMBER"]].copy()
        long_df.rename(columns={"DATE": "date", "SLOT": "slot", "NUMBER": "number"}, inplace=True)
    else:
        # Case 2: wide format with slot columns
        slots = ["FRBD", "GZBD", "GALI", "DSWR"]
        missing_slots = [col for col in slots if col not in df_raw.columns]
        if missing_slots:
            raise ValueError(f"Missing expected slot columns: {missing_slots}")

        wide_df = df_raw[["DATE"] + slots].copy()
        wide_df.rename(columns={"DATE": "date"}, inplace=True)
        long_df = wide_df.melt(
            id_vars=["date"],
            value_vars=slots,
            var_name="slot_name",
            value_name="number",
        )
        long_df["slot"] = long_df["slot_name"].map(SLOT_MAP)

    long_df["date"] = pd.to_datetime(long_df["date"], errors="coerce")
    long_df["slot"] = pd.to_numeric(long_df["slot"], errors="coerce")
    long_df["number"] = pd.to_numeric(long_df["number"], errors="coerce")

    long_df = long_df.dropna(subset=["date", "slot", "number"])
    if long_df.empty:
        print("❌ No valid entries after cleaning")
        return pd.DataFrame(columns=["date", "slot", "number"])

    long_df["slot"] = long_df["slot"].astype(int)
    long_df["number"] = long_df["number"].astype(int) % 100

    long_df = long_df[["date", "slot", "number"]].sort_values(["date", "slot"]).reset_index(drop=True)

    if long_df.empty:
        print("❌ No valid numeric entries after cleaning")

    return long_df
