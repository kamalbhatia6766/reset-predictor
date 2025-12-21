"""
Simple daily runner that executes SCR9 aggregation and produces a P&L snapshot.

Steps:
1) Run SCR9 (aggregates SCR1–SCR8) to save a shortlist.
2) Attach the latest result date to the shortlist and compute P&L windows.
3) Print a compact report (day/7d/month/cumulative) for quick sharing.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

from bet_pnl_tracker import PnLConfig, compute_pnl_report, render_compact_report
from quant_data_core import load_results_dataframe

SLOT_NAME_MAP = {"FRBD": 1, "GZBD": 2, "GALI": 3, "DSWR": 4}


def _run_scr9(top: int, per_slot_limit: int, output_dir: Path) -> None:
    cmd = [
        sys.executable,
        "deepseek_scr9.py",
        "--top",
        str(top),
        "--output-dir",
        str(output_dir),
        "--per-slot-limit",
        str(per_slot_limit),
    ]
    subprocess.run(cmd, check=True)


def _load_scr9_predictions(output_dir: Path, latest_date: pd.Timestamp) -> pd.DataFrame:
    """Load SCR9 shortlist and attach date/slot with weight columns intact."""

    shortlist_path = output_dir / "scr9_shortlist.csv"
    df = pd.read_csv(shortlist_path)
    df["date"] = pd.to_datetime(latest_date).date()
    df["slot"] = df["slot"].map(SLOT_NAME_MAP)

    keep_cols = [
        "date",
        "slot",
        "number",
        "tier",
        "in_top",
        "rank",
        "votes",
        "score",
    ]

    available = [c for c in keep_cols if c in df.columns]
    return df[available]


def main():
    parser = argparse.ArgumentParser(description="Run SCR9 and generate P&L report")
    parser.add_argument("--top", type=int, default=15, help="Top N for shortlist export")
    parser.add_argument(
        "--per-slot-limit", type=int, default=30, help="Cap per-script numbers per slot"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("predictions/deepseek_scr9"),
        help="Where SCR9 writes its shortlist",
    )
    args = parser.parse_args()

    results_df = load_results_dataframe()
    latest_date = pd.to_datetime(results_df["DATE"].max())

    _run_scr9(args.top, args.per_slot_limit, args.output_dir)
    preds = _load_scr9_predictions(args.output_dir, latest_date)

    report = compute_pnl_report(preds, cfg=PnLConfig())

    print(render_compact_report(report))


if __name__ == "__main__":
    main()
