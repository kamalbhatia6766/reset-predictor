"""
Minimal console runner that executes SCR1–SCR9 quietly, then prints a concise
summary (shortlist path, strongest Andar/Bahar digits) plus a compact Bet PnL
snapshot with layer performance metrics.
"""
from __future__ import annotations

import argparse  # NEW: For command-line argument parsing
import calendar
import datetime as dt
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import warnings

import pandas as pd
import numpy as np  # NEW: For numeric operations

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

from bet_pnl_tracker import (
    MAX_PICKS_CAP_DEFAULT,
    PnLConfig,
    SLOT_NAME_MAP,
    build_effective_dates,
    compute_pnl_report,
    format_andar_bahar_gating,
    format_hero_weakest,
    format_tag_roi,
    format_topk_profit,
    format_rank_bucket_windows,
    format_cross_slot_hits,
    load_clean_bet_rows,
    load_prebuilt_metrics,
    prebuilt_metrics_status,
    rebuild_prebuilt_metrics,
    render_compact_report,
)
from quant_data_core import load_results_dataframe
from regime_state_helper import compute_ab_gate_snapshot, compute_ab_gate_snapshot_metrics

SCRIPT_ORDER = [
    "deepseek_scr1.py",
    "deepseek_scr2.py",
    "deepseek_scr3.py",
    "deepseek_scr4.py",
    "deepseek_scr5.py",
    "deepseek_scr6.py",
    "deepseek_scr7.py",
    "deepseek_scr8.py",
    "deepseek_scr9.py",
]

OUTPUT_DIR = Path("predictions/deepseek_scr9")
HISTORY_PATH = OUTPUT_DIR / "scr9_shortlist_history.csv"
LOG_PATH = Path("logs/run_minimal.log")
SLOT_NAME_TO_ID = {v: k for k, v in SLOT_NAME_MAP.items()}
PREBUILT_DIR = Path("reports/prebuilt_metrics")
DEFAULT_WINDOW_DAYS = 90
GIT_FALLBACK_PATH = r"C:\Program Files\Git\bin\git.exe"
AUTO_PUSH_ENABLED = True


@dataclass
class DayCalc:
    date: dt.date
    cutoff: dt.date
    gates: Dict[int, bool]
    main_stake: float
    main_return: float
    main_pnl: float
    ab_stake: float
    ab_return: float
    ab_pnl: float
    total_stake: float
    total_pnl: float
    hits_per_slot: Dict[str, int]

# NEW: Add here after imports but before functions
def parse_ddmmyy(value: str) -> dt.date:
    """Parse date strings in DD-MM-YY format to date objects."""
    raw = value.strip()
    for fmt in ("%d-%m-%y",):
        try:
            return dt.datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    raise ValueError("Invalid date format. Use DD-MM-YY (e.g., 20-07-25).")


def _format_ddmmyy(value: dt.date) -> str:
    return value.strftime("%d-%m-%y")

def _resolve_ab_cutoff_date(prediction_date: dt.date, ab_cutoff: str) -> dt.date:
    if ab_cutoff == "same":
        return prediction_date
    return prediction_date - dt.timedelta(days=1)


def _is_valid_date(value: object) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except Exception:
        pass
    return isinstance(value, dt.date)


def parse_date_range(start_str: str, end_str: str) -> Tuple[dt.date, dt.date]:
    """Parse date strings in DD-MM-YY format to date objects."""
    start_date = parse_ddmmyy(start_str)
    end_date = parse_ddmmyy(end_str)
    return start_date, end_date

# NEW: Enhanced backtest function with date range
def _generate_daily_report(
    prediction_date: dt.date,
    shortlist: pd.DataFrame,
    trimmed_notes: List[str],
    k_auto_map: Optional[Dict[int, int]] = None,
    gate_snapshot: Optional[Dict[int, bool]] = None,
    cutoff_date: Optional[dt.date] = None,
) -> None:
    """Generate and save daily report for a specific date."""
    report_lines = []
    
    report_lines.append(f"{'='*60}")
    report_lines.append(f"DAILY PREDICTION REPORT - {prediction_date.strftime('%d-%m-%y')}")
    report_lines.append(f"{'='*60}")
    
    # Add strongest candidates
    candidates = _strongest_candidates(shortlist)
    report_lines.append("\n✅ STRONGEST ANDAR/BAHAR DIGITS")
    for slot_name, number in candidates:
        tens = number // 10
        ones = number % 10
        report_lines.append(f"{slot_name} → {number:02d} (tens:{tens}, ones:{ones})")
    
    # Add final bet numbers
    report_lines.append("\n📊 FINAL BET NUMBERS")
    bet_lines = _slot_bet_lines(shortlist, trimmed_notes)
    for line in bet_lines:
        report_lines.append(line)
    
    # Add K-AUTO information if available
    if k_auto_map and trimmed_notes:
        report_lines.append("\n🔧 K-AUTO LIMITS APPLIED")
        for note in trimmed_notes:
            report_lines.append(f"  {note}")
    
    # Calculate expected stake
    cfg = PnLConfig()
    total_picks = len(shortlist)
    expected_stake = total_picks * cfg.cost_per_unit
    report_lines.append(f"\n💰 BETTING SUMMARY")
    report_lines.append(f"Total Picks: {total_picks}")
    report_lines.append(f"Expected Stake: ₹{expected_stake:.0f}")
    report_lines.append(f"Cost per Pick: ₹{cfg.cost_per_unit:.0f}")
    report_lines.append(f"Payout on Hit: ₹{cfg.payout_per_unit:.0f}")

    cutoff = cutoff_date or (prediction_date - dt.timedelta(days=1))
    gate_snapshot = gate_snapshot or compute_ab_gate_snapshot(cutoff)
    gate_parts = []
    for slot_id in sorted(SLOT_NAME_MAP.keys()):
        slot_name = SLOT_NAME_MAP[slot_id]
        gate_parts.append(f"{slot_name}={'ON' if gate_snapshot.get(slot_id, False) else 'OFF'}")
    report_lines.append(
        f"\n🔒 Gate for next day (cutoff={cutoff.isoformat()}): "
        + ", ".join(gate_parts)
    )
    
    # Add timestamp
    report_lines.append(f"\n📅 Generated on: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Save to file
    report_dir = OUTPUT_DIR / prediction_date.strftime("%Y-%m-%d")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "daily_prediction_report.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

def run_backtest_date_range(
    start_date: dt.date,
    end_date: dt.date,
    auto_generate_missing: bool = True,
    ab_cutoff: str = "same",
    scr_timeout: int = 300,
    scr_retries: int = 1,
) -> None:
    """
    Run backtest for a specific date range.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"BACKTEST MODE: Date Range")
    logger.info(f"Window: {start_date.strftime('%d-%m-%y')} → {end_date.strftime('%d-%m-%y')}")
    logger.info(f"Auto-generate missing predictions: {'ON' if auto_generate_missing else 'OFF'}")
    logger.info(f"AB cutoff: {ab_cutoff}")
    logger.info(f"{'='*60}")
    
    # Load results data
    logger.debug("Loading results dataframe...")
    results_df = load_results_dataframe()
    results_df["DATE"] = pd.to_datetime(results_df["DATE"], errors="coerce").dt.date
    logger.debug(f"Loaded {len(results_df)} rows from results file")
    
    # Filter out month-end dates
    before_filter = len(results_df)
    results_df = results_df[~results_df["DATE"].apply(_is_month_end)]
    after_filter = len(results_df)
    if before_filter != after_filter:
        logger.debug(f"Filtered out {before_filter - after_filter} month-end dates")
    
    # Filter out rows with NaN in slot columns
    slot_columns = list(SLOT_NAME_TO_ID.keys())
    before_nan_filter = len(results_df)
    results_df = results_df.dropna(subset=slot_columns)
    after_nan_filter = len(results_df)
    if before_nan_filter != after_nan_filter:
        logger.debug(f"Filtered out {before_nan_filter - after_nan_filter} rows with NaN slots")
    
    if results_df.empty:
        logger.error("ERROR: No usable result dates available")
        return
    
    # Filter by date range
    mask = (results_df["DATE"] >= start_date) & (results_df["DATE"] <= end_date)
    date_range_df = results_df[mask]
    logger.debug(f"Filtered to date range: {len(date_range_df)} rows")
    
    if date_range_df.empty:
        logger.error(f"ERROR: No results available in date range {start_date.strftime('%d-%m-%y')} to {end_date.strftime('%d-%m-%y')}")
        return
    
    # Get available dates in range
    available_dates = sorted(date_range_df["DATE"].unique())
    total_days = len(available_dates)
    
    logger.info(f"Backtest window: {start_date.strftime('%d-%m-%y')} → {end_date.strftime('%d-%m-%y')}")
    logger.info(f"Days to process: {total_days}")
    
    # Try to load K-AUTO map from historical metrics
    cfg = PnLConfig()
    # Initialize with default values for all slots
    k_auto_map = {slot_id: MAX_PICKS_CAP_DEFAULT for slot_id in SLOT_NAME_MAP.keys()}
    
    try:
        logger.debug("Loading K-AUTO map from historical metrics...")
        # Get effective dates for metrics calculation
        effective_dates = build_effective_dates(start_date, end_date, available_dates=available_dates)
        logger.debug(f"Effective dates for metrics: {len(effective_dates)} dates")
        
        # Load bet rows to calculate K-AUTO
        bet_rows = load_clean_bet_rows(start_date, end_date, cfg)
        logger.debug(f"Loaded {len(bet_rows)} bet rows for K-AUTO calculation")
        
        if not bet_rows.empty:
            _, loaded_k_auto_map = format_rank_bucket_windows(bet_rows, effective_dates)
            if loaded_k_auto_map:
                k_auto_map.update(loaded_k_auto_map)  # Update with loaded values
                logger.info(f"K-AUTO limits loaded: {', '.join([f'{SLOT_NAME_MAP[sid]}={cap}' for sid, cap in k_auto_map.items()])}")
        else:
            logger.warning(f"⚠️  No bet history available for K-AUTO calculation, using default cap of {MAX_PICKS_CAP_DEFAULT}")
    except Exception as e:
        logger.warning(f"⚠️  Warning: Could not load K-AUTO map: {e}")
        logger.warning(f"    Using default cap of {MAX_PICKS_CAP_DEFAULT} for all slots")
    
    logger.info("-" * 60)
    
    project_root = Path(__file__).resolve().parent

    all_predictions: List[pd.DataFrame] = []
    gate_by_day: Dict[dt.date, Dict[int, bool]] = {}
    cutoff_by_day: Dict[dt.date, dt.date] = {}

    # Process each day
    for i, current_date in enumerate(available_dates):
        cutoff_date = _resolve_ab_cutoff_date(current_date, ab_cutoff)
        gate_snapshot = compute_ab_gate_snapshot(cutoff_date)
        gate_by_day[current_date] = gate_snapshot
        cutoff_by_day[current_date] = cutoff_date
        logger.info(
            "AB Gate Snapshot (cutoff=%s): %s",
            cutoff_date.isoformat(),
            _format_gate_flags(gate_snapshot),
        )
        if logger.isEnabledFor(logging.DEBUG):
            _, roi_map = compute_ab_gate_snapshot_metrics(cutoff_date)
            prefix = f"AB Gate Inputs (cutoff={cutoff_date.isoformat()})"
            for idx, slot_id in enumerate(sorted(SLOT_NAME_MAP.keys())):
                slot_name = SLOT_NAME_MAP[slot_id]
                roi_7 = roi_map.get(slot_id, {}).get("7d")
                roi_30 = roi_map.get(slot_id, {}).get("30d")
                roi_all = roi_map.get(slot_id, {}).get("all")
                def _fmt_roi(value: Optional[float]) -> str:
                    return "n/a" if value is None else f"{value:+.0%}"

                gate_state = "ON" if gate_by_day[current_date].get(slot_id, False) else "OFF"
                line_prefix = prefix if idx == 0 else " " * len(prefix)
                logger.debug(
                    "%s  %s: 7d=%s 30d=%s all=%s → %s",
                    line_prefix,
                    slot_name,
                    _fmt_roi(roi_7),
                    _fmt_roi(roi_30),
                    _fmt_roi(roi_all),
                    gate_state,
                )

        logger.info(f"Processing {current_date.strftime('%d-%m-%y')} ({i+1}/{total_days})...")
        logger.debug(f"  Current date: {current_date}")

        shortlist = _load_or_generate_shortlist(
            current_date,
            k_auto_map,
            project_root,
            auto_generate_missing,
            ab_cutoff,
            scr_timeout=scr_timeout,
            scr_retries=scr_retries,
        )
        if shortlist is None:
            continue

        preds = _prepare_predictions(shortlist, current_date)
        logger.debug(f"  Prepared {len(preds)} predictions")

        if preds.empty:
            logger.warning(f"  SKIP: No predictions for {current_date.strftime('%d-%m-%y')}")
            continue

        all_predictions.append(preds)

    if not all_predictions:
        logger.info("No results to report")
        return

    predictions_df = pd.concat(all_predictions, ignore_index=True)
    pnl_report = compute_pnl_report(predictions_df, cfg, gate_by_day=gate_by_day)

    merged = pnl_report.merged
    digit_pnl = pnl_report.digit_pnl

    if merged.empty:
        logger.info("No results to report")
        return

    daily_dates = sorted(set(merged["date"].unique()) | set(digit_pnl["date"].unique()))
    day_calcs: List[DayCalc] = []
    slot_digit_hits = pnl_report.slot_digit_hits
    for report_date in daily_dates:
        main_day = merged[merged["date"] == report_date][["cost", "payout", "pnl"]].sum()
        ab_day = digit_pnl[digit_pnl["date"] == report_date][["cost", "payout", "pnl"]].sum()

        main_stake = float(main_day.get("cost", 0.0))
        main_return = float(main_day.get("payout", 0.0))
        main_pnl = float(main_day.get("pnl", 0.0))
        ab_stake = float(ab_day.get("cost", 0.0))
        ab_return = float(ab_day.get("payout", 0.0))
        ab_pnl = float(ab_day.get("pnl", 0.0))

        gate_snapshot = gate_by_day.get(report_date, {})
        cutoff_date = cutoff_by_day.get(report_date, report_date - dt.timedelta(days=1))
        hits_by_slot = (
            merged[(merged["date"] == report_date) & (merged["hit"])]
            .groupby("slot_name")["hit"]
            .sum()
            .to_dict()
        )
        for slot_name in SLOT_NAME_TO_ID.keys():
            hits_by_slot.setdefault(slot_name, 0)

        total_stake = main_stake + ab_stake
        total_pnl = main_pnl + ab_pnl
        expected_total = (main_return - main_stake) + (ab_return - ab_stake)
        if not np.isclose(total_pnl, expected_total, atol=0.01):
            gate_line = _format_gate_flags(gate_snapshot)
            chosen_digits = _format_ab_digits(slot_digit_hits, report_date)
            logger.warning(
                "SELF-TEST FAIL %s | cutoff=%s | gates=%s | AB stake=₹%.0f | AB return=₹%.0f | AB digits=%s",
                _format_ddmmyy(report_date),
                cutoff_date.isoformat(),
                gate_line,
                ab_stake,
                ab_return,
                chosen_digits,
            )
        elif logger.isEnabledFor(logging.DEBUG):
            logger.debug("SELF-TEST PASS %s", _format_ddmmyy(report_date))

        day_calcs.append(
            DayCalc(
                date=report_date,
                cutoff=cutoff_date,
                gates=gate_snapshot,
                main_stake=main_stake,
                main_return=main_return,
                main_pnl=main_pnl,
                ab_stake=ab_stake,
                ab_return=ab_return,
                ab_pnl=ab_pnl,
                total_stake=total_stake,
                total_pnl=total_pnl,
                hits_per_slot=hits_by_slot,
            )
        )

    daily_results = pd.DataFrame(
        [
            {
                "date": day.date,
                "stake": day.main_stake,
                "return": day.main_return,
                "pnl": day.main_pnl,
                "ab_stake": day.ab_stake,
                "ab_return": day.ab_return,
                "ab_pnl": day.ab_pnl,
                "total_stake": day.total_stake,
                "total_pnl": day.total_pnl,
            }
            for day in day_calcs
        ]
    ).sort_values("date")

    daily_results["roi"] = np.where(
        daily_results["stake"] > 0, daily_results["pnl"] / daily_results["stake"] * 100, 0
    )
    daily_results["total_roi"] = np.where(
        daily_results["total_stake"] > 0,
        daily_results["total_pnl"] / daily_results["total_stake"] * 100,
        0,
    )

    slot_base = pnl_report.slot_totals.rename(
        columns={"cost": "stake", "payout": "return", "pnl": "pnl"}
    )
    slot_ab = (
        digit_pnl.groupby(["slot", "slot_name"])[["cost", "payout", "pnl"]]
        .sum()
        .reset_index()
        .rename(columns={"cost": "ab_stake", "payout": "ab_return", "pnl": "ab_pnl"})
    )
    slot_totals = pd.merge(slot_base, slot_ab, on=["slot", "slot_name"], how="left").fillna(0)
    slot_hits = merged.groupby(["slot", "slot_name"])["hit"].sum().reset_index(name="hits")
    slot_totals = pd.merge(slot_totals, slot_hits, on=["slot", "slot_name"], how="left").fillna(0)
    slot_totals["roi"] = np.where(
        slot_totals["stake"] > 0, slot_totals["pnl"] / slot_totals["stake"] * 100, 0
    )
    slot_totals["total_pnl"] = slot_totals["pnl"] + slot_totals["ab_pnl"]
    slot_totals["total_stake"] = slot_totals["stake"] + slot_totals["ab_stake"]
    slot_totals["total_roi"] = np.where(
        slot_totals["total_stake"] > 0,
        slot_totals["total_pnl"] / slot_totals["total_stake"] * 100,
        0,
    )

    total_hits = sum(sum(day.hits_per_slot.values()) for day in day_calcs)
    overall_totals = {
        "stake": float(sum(day.main_stake for day in day_calcs)),
        "return": float(sum(day.main_return for day in day_calcs)),
        "pnl": float(sum(day.main_pnl for day in day_calcs)),
        "ab_stake": float(sum(day.ab_stake for day in day_calcs)),
        "ab_return": float(sum(day.ab_return for day in day_calcs)),
        "ab_pnl": float(sum(day.ab_pnl for day in day_calcs)),
        "total_stake": float(sum(day.total_stake for day in day_calcs)),
        "total_pnl": float(sum(day.total_pnl for day in day_calcs)),
        "hits": int(total_hits),
    }
    
    logger.info(f"\n{'='*60}")
    logger.info("BACKTEST COMPLETE")
    logger.info(f"{'='*60}")
    
    # Calculate and display summary
    if daily_results.empty:
        logger.info("No results to report")
        return
    
    logger.debug(f"Overall totals: {overall_totals}")

    logger.info("\nDAY-BY-DAY SUMMARY:")
    logger.info("Date        Stake    Return   P&L      ROI%     ABRet  ABPnL  TotPnL  TotROI")
    logger.info("----        -----    ------   ----     ----     -----  -----  ------  ------")
    for day in day_calcs:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "AB Stake Day=₹%.0f | AB Return=₹%.0f | AB P&L=%+.0f",
                day.ab_stake,
                day.ab_return,
                day.ab_pnl,
            )
        logger.info(
            f"{day.date.strftime('%d-%m-%y'):<11}  {day.main_stake:>6.0f}  "
            f"{day.main_return:>8.0f}  {day.main_pnl:>+7.0f}  "
            f"{(day.main_pnl / day.main_stake * 100) if day.main_stake > 0 else 0:>+6.1f}%  "
            f"{day.ab_return:>6.0f}  {day.ab_pnl:>+6.0f}  {day.total_pnl:>+7.0f}  "
            f"{(day.total_pnl / day.total_stake * 100) if day.total_stake > 0 else 0:>+6.1f}%"
        )
    
    # Overall summary
    total_roi = (overall_totals["pnl"] / overall_totals["stake"] * 100) if overall_totals["stake"] > 0 else 0
    total_ab_roi = (
        overall_totals["ab_pnl"] / overall_totals["ab_stake"] * 100
        if overall_totals["ab_stake"] > 0
        else 0
    )
    total_combined_roi = (
        overall_totals["total_pnl"] / overall_totals["total_stake"] * 100
        if overall_totals["total_stake"] > 0
        else 0
    )
    hit_rate = (overall_totals["hits"] / (len(daily_results) * 4) * 100) if not daily_results.empty else 0
    
    logger.info(f"\nOVERALL SUMMARY ({len(daily_results)} days):")
    logger.info(f"Total Stake:    ₹{overall_totals['stake']:.0f}")
    logger.info(f"Total Return:   ₹{overall_totals['return']:.0f}")
    logger.info(f"Total P&L:      ₹{overall_totals['pnl']:+.0f}")
    logger.info(f"Overall ROI:    {total_roi:+.1f}%")
    logger.info(f"AB Stake:       ₹{overall_totals['ab_stake']:.0f}")
    logger.info(f"AB Return:      ₹{overall_totals['ab_return']:.0f}")
    logger.info(f"AB P&L:         ₹{overall_totals['ab_pnl']:+.0f}")
    logger.info(f"AB ROI:         {total_ab_roi:+.1f}%")
    logger.info(f"Total P&L:      ₹{overall_totals['total_pnl']:+.0f}")
    logger.info(f"Total ROI:      {total_combined_roi:+.1f}%")
    logger.info(f"Total Hits:     {overall_totals['hits']}")
    logger.info(f"Hit Rate:       {hit_rate:.1f}%")
    
    # Per-slot summary
    logger.info(f"\nPER-SLOT SUMMARY:")
    logger.info("Slot    Stake    Return   P&L      ROI%     ABRet  ABPnL  TotPnL  TotROI   Hits")
    logger.info("----    -----    ------   ----     ----     -----  -----  ------  ------  ----")
    for _, totals in slot_totals.sort_values("slot").iterrows():
        if totals["stake"] > 0:
            logger.info(
                f"{totals['slot_name']:<6}  {totals['stake']:>6.0f}  {totals['return']:>8.0f}  "
                f"{totals['pnl']:>+7.0f}  {totals['roi']:>+6.1f}%  {totals['ab_return']:>6.0f}  "
                f"{totals['ab_pnl']:>+6.0f}  {totals['total_pnl']:>+7.0f}  "
                f"{totals['total_roi']:>+6.1f}%  {int(totals['hits']):>4}"
            )
    
    # Calculate 7D and 30D ROI within the backtest window
    if len(daily_results) >= 7:
        last_7_days = daily_results.tail(7)
        stake_7d = float(last_7_days["stake"].sum())
        pnl_7d = float(last_7_days["pnl"].sum())
        roi_7d = (pnl_7d / stake_7d * 100) if stake_7d > 0 else 0
        logger.info(f"\n7-Day ROI:      {roi_7d:+.1f}%")
    else:
        logger.info(f"\n7-Day ROI:      INSUFFICIENT DATA")
    
    if len(daily_results) >= 30:
        last_30_days = daily_results.tail(30)
        stake_30d = float(last_30_days["stake"].sum())
        pnl_30d = float(last_30_days["pnl"].sum())
        roi_30d = (pnl_30d / stake_30d * 100) if stake_30d > 0 else 0
        logger.info(f"30-Day ROI:     {roi_30d:+.1f}%")
    else:
        logger.info(f"30-Day ROI:     INSUFFICIENT DATA")
    
    # Save report to file
    start_str = start_date.strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")
    report_path = Path(f"reports/backtest_{start_str}_{end_str}.txt")
    
    report_lines = [
        f"{'='*60}",
        f"BACKTEST REPORT: Date Range {start_date.strftime('%d-%m-%y')} to {end_date.strftime('%d-%m-%y')}",
        f"Days processed: {len(daily_results)}",
        f"{'='*60}\n",
        f"OVERALL SUMMARY:",
        f"Total Stake:    ₹{overall_totals['stake']:.0f}",
        f"Total Return:   ₹{overall_totals['return']:.0f}",
        f"Total P&L:      ₹{overall_totals['pnl']:+.0f}",
        f"Overall ROI:    {total_roi:+.1f}%",
        f"AB Stake:       ₹{overall_totals['ab_stake']:.0f}",
        f"AB Return:      ₹{overall_totals['ab_return']:.0f}",
        f"AB P&L:         ₹{overall_totals['ab_pnl']:+.0f}",
        f"AB ROI:         {total_ab_roi:+.1f}%",
        f"Total P&L:      ₹{overall_totals['total_pnl']:+.0f}",
        f"Total ROI:      {total_combined_roi:+.1f}%",
        f"Total Hits:     {overall_totals['hits']}",
        f"Hit Rate:       {hit_rate:.1f}%\n",
        f"PER-SLOT SUMMARY:",
        "Slot    Stake    Return   P&L      ROI%     ABRet  ABPnL  TotPnL  TotROI   Hits",
        "----    -----    ------   ----     ----     -----  -----  ------  ------  ----"
    ]
    
    for _, totals in slot_totals.sort_values("slot").iterrows():
        if totals["stake"] > 0:
            report_lines.append(
                f"{totals['slot_name']:<6}  {totals['stake']:>6.0f}  {totals['return']:>8.0f}  "
                f"{totals['pnl']:>+7.0f}  {totals['roi']:>+6.1f}%  {totals['ab_return']:>6.0f}  "
                f"{totals['ab_pnl']:>+6.0f}  {totals['total_pnl']:>+7.0f}  "
                f"{totals['total_roi']:>+6.1f}%  {int(totals['hits']):>4}"
            )
    
    report_lines.append("")
    if len(daily_results) >= 7:
        report_lines.append(f"7-Day ROI:      {roi_7d:+.1f}%")
    else:
        report_lines.append(f"7-Day ROI:      INSUFFICIENT DATA")
    
    if len(daily_results) >= 30:
        report_lines.append(f"30-Day ROI:     {roi_30d:+.1f}%")
    else:
        report_lines.append(f"30-Day ROI:     INSUFFICIENT DATA")
    
    report_lines.append(f"\n{'='*60}")
    report_lines.append("DAILY DETAILS:")
    report_lines.append("Date        Stake    Return   P&L      ROI%     ABRet  ABPnL  TotPnL  TotROI")
    report_lines.append("----        -----    ------   ----     ----     -----  -----  ------  ------")
    
    for _, result in daily_results.iterrows():
        report_lines.append(
            f"{result['date'].strftime('%d-%m-%y'):<11}  {result['stake']:>6.0f}  "
            f"{result['return']:>8.0f}  {result['pnl']:>+7.0f}  "
            f"{result['roi']:>+6.1f}%  {result['ab_return']:>6.0f}  "
            f"{result['ab_pnl']:>+6.0f}  {result['total_pnl']:>+7.0f}  "
            f"{result['total_roi']:>+6.1f}%"
        )
    
    _write_text(report_path, "\n".join(report_lines))
    logger.info(f"\nReport saved to: {report_path}")
    _autopush_if_needed(f"Backtest reports update {start_str}-{end_str}")

# Helper functions for the script
def _rebuild_metrics_if_needed() -> None:
    """Rebuild metrics from historical data."""
    try:
        logger.debug("Rebuilding metrics from historical data...")
        # Load results to get date range
        results_df = load_results_dataframe()
        results_df["DATE"] = pd.to_datetime(results_df["DATE"], errors="coerce").dt.date
        slot_columns = list(SLOT_NAME_TO_ID.keys())
        results_df = results_df.dropna(subset=slot_columns)
        
        all_dates = sorted([d for d in results_df["DATE"].unique() if d is not None])
        if not all_dates:
            raise ValueError("No dates available for metrics rebuild")
        
        # Use last 90 days or all available data
        if len(all_dates) >= 90:
            metrics_start = all_dates[-90]
        else:
            metrics_start = all_dates[0]
        metrics_end = all_dates[-1]
        
        logger.debug(f"Metrics rebuild date range: {metrics_start} to {metrics_end}")
        
        cfg = PnLConfig()
        rebuild_prebuilt_metrics(metrics_start, metrics_end, cfg)
        logger.info("✅ Metrics rebuild complete")
    except Exception as e:
        logger.warning(f"⚠️  Warning: Failed to rebuild metrics: {e}")


def _metrics_needs_rebuild() -> tuple[bool, str]:
    latest_result_date = _get_latest_result_date()
    if not _is_valid_date(latest_result_date):
        return True, "latest result date missing"
    return prebuilt_metrics_status(latest_result_date, PREBUILT_DIR)


def _maybe_rebuild_metrics(log_skip: bool = True) -> bool:
    needs_rebuild, reason = _metrics_needs_rebuild()
    if not needs_rebuild:
        if log_skip:
            logger.info("NO METRICS REBUILD")
        logger.debug("Metrics rebuild skipped: %s", reason)
        return False
    logger.info("\n🔄 Rebuilding metrics from historical data...")
    _rebuild_metrics_if_needed()
    return True


def _git_available() -> bool:
    return _resolve_git_exe() is not None


def _resolve_git_exe() -> Optional[str]:
    git_exe = shutil.which("git")
    if git_exe:
        return git_exe
    fallback = Path(GIT_FALLBACK_PATH)
    if fallback.exists():
        return str(fallback)
    return None


def _git_repo_available() -> bool:
    git_exe = _resolve_git_exe()
    if not git_exe:
        return False
    try:
        subprocess.run([git_exe, "rev-parse", "--is-inside-work-tree"], check=True, capture_output=True, text=True)
        return True
    except Exception:
        return False


def _git_status_lines(paths: Optional[List[str]] = None) -> List[str]:
    git_exe = _resolve_git_exe()
    if not git_exe:
        raise FileNotFoundError("git not available")
    cmd = [git_exe, "status", "--porcelain"]
    if paths:
        cmd.extend(paths)
    result = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.strip()
    return [line for line in result.splitlines() if line.strip()]


def _should_autopush() -> bool:
    try:
        return bool(_git_status_lines(["predictions", "reports"]))
    except Exception:
        return False


def _run_git_autopush() -> None:
    script_path = Path("tools") / "git_autopush.bat"
    if not script_path.exists():
        logger.warning("Auto-push script missing: %s", script_path)
        return
    try:
        logger.info("Running auto-push: %s", script_path)
        subprocess.run([str(script_path)], check=False, capture_output=True, text=True)
    except Exception as exc:
        logger.warning("Auto-push failed: %s", exc)


def _check_git_status(explicit: bool = False) -> None:
    """Print a compact git status summary. Never raises."""
    if not _git_available() or not _git_repo_available():
        if explicit:
            logger.info("Detected: Git=NO (skipping status/push)")
        logger.warning("Git not available; skipping git status/push")
        return
    git_exe = _resolve_git_exe()
    if not git_exe:
        logger.warning("Git not available; skipping git status/push")
        return

    try:
        version = subprocess.run(
            [git_exe, "--version"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        branch = subprocess.run(
            [git_exe, "rev-parse", "--abbrev-ref", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status_lines = _git_status_lines()
    except subprocess.CalledProcessError:
        logger.warning("Git not available; skipping git status/push")
        return

    if explicit:
        logger.info(version)
        logger.info(f"Detected: Git=YES (branch={branch}, changed={len(status_lines)})")
        return

    logger.info("🔍 GIT STATUS")
    logger.info(version)
    logger.info(f"Branch: {branch}")
    logger.info(f"Changed files: {len(status_lines)}")
    logger.info(f"Auto-push: {'would attempt' if _should_autopush() else 'not needed'}")


def safe_autopush(message: str) -> None:
    """Attempt git pull/add/commit/push with warnings instead of failures."""
    git_exe = _resolve_git_exe()
    if not git_exe or not _git_repo_available():
        logger.warning("Git not available; skipping git status/push")
        return

    try:
        subprocess.run(
            [git_exe, "pull", "--rebase", "--autostash"],
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception as exc:
        logger.warning(f"⚠️  Git pull failed: {exc}")

    try:
        subprocess.run([git_exe, "add", "-A"], check=True, capture_output=True, text=True)
    except Exception as exc:
        logger.warning(f"⚠️  Git add failed: {exc}")
        return

    try:
        commit_result = subprocess.run(
            [git_exe, "commit", "-m", message],
            check=False,
            capture_output=True,
            text=True,
        )
        if commit_result.returncode not in (0, 1):
            logger.warning(f"⚠️  Git commit failed: {commit_result.stderr.strip()}")
    except Exception as exc:
        logger.warning(f"⚠️  Git commit failed: {exc}")

    try:
        subprocess.run([git_exe, "push"], check=True, capture_output=True, text=True)
    except Exception as exc:
        logger.warning(f"⚠️  Git push failed: {exc}")


def _autopush_if_needed(message: str) -> None:
    if not AUTO_PUSH_ENABLED:
        return
    if _should_autopush():
        _run_git_autopush()
        return
    logger.info("No changes to push.")


def _is_month_end(d: dt.date) -> bool:
    """Check if a date is the last day of the month."""
    next_day = d + dt.timedelta(days=1)
    return next_day.month != d.month

def _latest_non_month_end_date(dates: List[dt.date]) -> dt.date:
    """Find the latest date that is not a month-end."""
    valid_dates = [d for d in dates if not _is_month_end(d)]
    if not valid_dates:
        raise ValueError("No non-month-end dates available")
    return max(valid_dates)

def _next_non_month_end(start_date: dt.date) -> dt.date:
    """Find the next date that is not a month-end."""
    current = start_date
    while _is_month_end(current):
        current += dt.timedelta(days=1)
    return current

def _load_shortlist() -> pd.DataFrame:
    """Load the shortlist generated by SCR9."""
    shortlist_path = OUTPUT_DIR / "scr9_shortlist.csv"
    if not shortlist_path.exists():
        raise FileNotFoundError(f"Shortlist not found: {shortlist_path}")
    return pd.read_csv(shortlist_path)

def _load_shortlist_for_date(target_date: dt.date) -> pd.DataFrame:
    """Load the saved shortlist for a specific historical date.
    
    This is used in backtest mode to load predictions that were actually
    made on that date, rather than generating fresh predictions.
    
    Expected file structure:
        predictions/deepseek_scr9/YYYY-MM-DD/shortlist.xlsx (preferred)
        predictions/deepseek_scr9/YYYY-MM-DD/shortlist.csv (fallback)
    
    Priority: Excel file is checked first, then CSV if Excel not found.
    
    Args:
        target_date: The date for which to load saved predictions
        
    Returns:
        DataFrame containing the saved predictions
        
    Raises:
        FileNotFoundError: If no saved predictions exist for the date
        Exception: If file is corrupted or cannot be read
    """
    date_dir = OUTPUT_DIR / target_date.strftime("%Y-%m-%d")
    xlsx_path = date_dir / "shortlist.xlsx"
    csv_path = date_dir / "shortlist.csv"
    legacy_xlsx_path = date_dir / "scr9_shortlist.xlsx"
    legacy_csv_path = date_dir / "scr9_shortlist.csv"
    
    # Try Excel first, then CSV
    if xlsx_path.exists():
        try:
            return pd.read_excel(xlsx_path)
        except Exception as e:
            raise Exception(f"Failed to read Excel file {xlsx_path}: {e}")
    elif csv_path.exists():
        try:
            return pd.read_csv(csv_path)
        except Exception as e:
            raise Exception(f"Failed to read CSV file {csv_path}: {e}")
    elif legacy_xlsx_path.exists():
        try:
            return pd.read_excel(legacy_xlsx_path)
        except Exception as e:
            raise Exception(f"Failed to read Excel file {legacy_xlsx_path}: {e}")
    elif legacy_csv_path.exists():
        try:
            return pd.read_csv(legacy_csv_path)
        except Exception as e:
            raise Exception(f"Failed to read CSV file {legacy_csv_path}: {e}")
    else:
        raise FileNotFoundError(
            f"No saved predictions found for {target_date.strftime('%d-%m-%y')}. "
            f"Expected at: {xlsx_path} or {csv_path}"
        )

def _write_scr2_error(
    project_root: Path,
    error_date: dt.date,
    returncode: object,
    stderr: str,
) -> None:
    error_dir = project_root / "predictions" / "deepseek_scr2"
    error_dir.mkdir(parents=True, exist_ok=True)
    stderr_tail = (stderr or "")[-2000:]
    error_path = error_dir / "_last_error.txt"
    with open(error_path, "w", encoding="utf-8") as handle:
        handle.write(
            "date: {date}\nreturncode: {code}\nstderr_tail:\n{tail}".format(
                date=error_date.strftime("%Y-%m-%d"),
                code=returncode,
                tail=stderr_tail,
            )
        )


def _run_scr2_with_retry(
    project_root: Path,
    env: Dict[str, str],
    *,
    timeout: int,
    retries: int,
    run_date: Optional[dt.date],
) -> None:
    attempts = max(retries, 0) + 1
    display_date = (run_date or dt.date.today()).strftime("%Y-%m-%d")
    logger.info(f"scr2: start for {display_date} (timeout={timeout}, retries={retries})")

    last_code: object = None
    last_stderr = ""
    for attempt in range(1, attempts + 1):
        try:
            result = subprocess.run(
                [sys.executable, str(project_root / "deepseek_scr2.py")],
                capture_output=True,
                text=True,
                cwd=project_root,
                env=env,
                timeout=timeout,
            )
            if result.returncode == 0:
                return
            last_code = result.returncode
            last_stderr = result.stderr or ""
            if attempt < attempts:
                logger.info(
                    "scr2: nonzero exit %s on attempt %s/%s ... retrying in 5s",
                    last_code,
                    attempt,
                    attempts,
                )
                time.sleep(5)
                continue
            break
        except subprocess.TimeoutExpired as exc:
            last_code = "timeout"
            last_stderr = exc.stderr or ""
            if attempt < attempts:
                logger.info(
                    "scr2: timeout after %ss on attempt %s/%s ... retrying in 5s",
                    timeout,
                    attempt,
                    attempts,
                )
                time.sleep(5)
                continue
            break
        except Exception as exc:
            last_code = "error"
            last_stderr = str(exc)
            if attempt < attempts:
                logger.info(
                    "scr2: error on attempt %s/%s ... retrying in 5s",
                    attempt,
                    attempts,
                )
                time.sleep(5)
                continue
            break

    if last_code == 3221225786:
        logger.info("scr2: terminated/keyboard interrupt (external kill)")
        reason = "terminated"
    elif last_code == "timeout":
        reason = "timeout"
    else:
        reason = "nonzero exit"

    logger.info(
        "scr2: SOFT-FAIL (code=%s, reason=%s) continuing without scr2.",
        last_code,
        reason,
    )
    _write_scr2_error(project_root, run_date or dt.date.today(), last_code, last_stderr)


def _run_scripts_quietly(
    project_root: Path,
    cutoff_date: Optional[dt.date] = None,
    *,
    scr_timeout: int = 300,
    scr_retries: int = 1,
    scr2_date: Optional[dt.date] = None,
) -> None:
    """Run all SCR scripts (1-9) quietly without printing output."""
    env = os.environ.copy()
    if cutoff_date is not None:
        env["PREDICTOR_CUTOFF_DATE"] = cutoff_date.strftime("%Y-%m-%d")
    else:
        env.pop("PREDICTOR_CUTOFF_DATE", None)
    for script in SCRIPT_ORDER:
        script_path = project_root / script
        if not script_path.exists():
            logger.warning(f"⚠️  Warning: Script not found: {script}")
            continue

        if script == "deepseek_scr2.py":
            _run_scr2_with_retry(
                project_root,
                env,
                timeout=scr_timeout,
                retries=scr_retries,
                run_date=scr2_date,
            )
            continue
        
        try:
            logger.debug(f"Running script: {script}")
            # Run the script quietly
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                cwd=project_root,
                env=env,
                timeout=300  # 5 minute timeout
            )
            if result.returncode != 0:
                logger.warning(f"⚠️  Warning: {script} failed with code {result.returncode}")
                logger.debug(f"  stderr: {result.stderr[:200]}")
        except subprocess.TimeoutExpired:
            logger.warning(f"⚠️  Warning: {script} timed out")
        except Exception as e:
            logger.warning(f"⚠️  Warning: Error running {script}: {e}")

def _strongest_candidates(shortlist: pd.DataFrame) -> List[Tuple[str, int]]:
    """Extract the strongest Andar/Bahar digit candidate for each slot."""
    candidates = []
    for slot_name in SLOT_NAME_TO_ID.keys():
        slot_data = shortlist[shortlist["slot"] == slot_name]
        if not slot_data.empty:
            # Get the top ranked number
            top_row = slot_data.iloc[0]
            number = int(top_row["number"])
            candidates.append((slot_name, number))
    return candidates

def _slot_bet_lines(shortlist: pd.DataFrame, trim_notes: List[str]) -> List[str]:
    """Format bet numbers per slot with pick counts."""
    lines = []
    for slot_name in SLOT_NAME_TO_ID.keys():
        slot_data = shortlist[shortlist["slot"] == slot_name]
        if not slot_data.empty:
            # Get top numbers for this slot (up to 10 for display)
            num_picks = len(slot_data)
            display_limit = min(10, num_picks)
            top_numbers = slot_data.head(display_limit)["number"].tolist()
            numbers_str = ", ".join([f"{int(n):02d}" for n in top_numbers])
            
            # Add ellipsis if more numbers exist
            suffix = f" ... ({num_picks} total)" if num_picks > display_limit else f" ({num_picks} total)"
            lines.append(f"{slot_name}: {numbers_str}{suffix}")
    return lines

def _apply_max_cap(shortlist: pd.DataFrame, k_auto_map: Optional[Dict[int, int]]) -> tuple[pd.DataFrame, List[str]]:
    """Apply K-AUTO cap to limit picks per slot based on historical performance."""
    if shortlist.empty:
        return shortlist, []
    
    # Handle None or empty k_auto_map
    if not k_auto_map:
        k_auto_map = {}
    
    trimmed = shortlist.copy()
    notes: List[str] = []
    
    logger.debug(f"Applying K-AUTO caps: {k_auto_map}")
    
    for slot_name in SLOT_NAME_TO_ID.keys():
        group = shortlist[shortlist["slot"] == slot_name]
        if group.empty:
            logger.debug(f"  {slot_name}: No predictions to cap")
            continue
        
        # Validate slot_id
        slot_id = SLOT_NAME_TO_ID.get(slot_name)
        if slot_id is None:
            notes.append(f"⚠️  Unknown slot name: {slot_name}")
            logger.warning(f"Unknown slot name: {slot_name}")
            continue
        
        cap = k_auto_map.get(slot_id, MAX_PICKS_CAP_DEFAULT)
        original_count = len(group)
        
        if len(group) <= cap:
            logger.debug(f"  {slot_name}: {original_count} picks ≤ cap ({cap}), no trimming needed")
            continue
        
        # Sort by rank (ascending) to keep top-ranked numbers
        # This preserves the original SCR9 ranking order
        ranked = group.sort_values("rank", ascending=True)
        keep = ranked.head(cap)
        trimmed = trimmed.drop(group.index.difference(keep.index))
        notes.append(f"{slot_name}: trimmed {len(group)}→{cap} (K-AUTO)")
        logger.debug(f"  {slot_name}: Trimmed {original_count} → {cap} picks (K-AUTO cap applied)")
    
    logger.debug(f"Total predictions after K-AUTO: {len(trimmed)}")
    return trimmed, notes

def _prepare_predictions(shortlist: pd.DataFrame, prediction_date: dt.date) -> pd.DataFrame:
    """Prepare predictions dataframe with date and slot information."""
    if shortlist.empty:
        return pd.DataFrame()
    
    # Add prediction date
    shortlist = shortlist.copy()
    shortlist["date"] = prediction_date

    # Preserve slot name and map to IDs
    shortlist["slot_name"] = shortlist["slot"]
    shortlist["slot"] = shortlist["slot"].map(SLOT_NAME_TO_ID)

    # Ensure number is integer
    shortlist["number"] = shortlist["number"].astype(int)
    
    return shortlist

def _format_gate_flags(gate_snapshot: Dict[int, bool]) -> str:
    gate_parts = []
    for slot_id in sorted(SLOT_NAME_MAP.keys()):
        slot_name = SLOT_NAME_MAP[slot_id]
        gate_parts.append(f"{slot_name}={'ON' if gate_snapshot.get(slot_id, False) else 'OFF'}")
    return ", ".join(gate_parts)

def _format_ab_digits(slot_digit_hits: pd.DataFrame, report_date: dt.date) -> str:
    if slot_digit_hits.empty:
        return "n/a"
    day_rows = slot_digit_hits[slot_digit_hits["date"] == report_date]
    if day_rows.empty:
        return "n/a"
    parts = []
    for slot_name in SLOT_NAME_TO_ID.keys():
        slot_row = day_rows[day_rows["slot_name"] == slot_name]
        if slot_row.empty:
            parts.append(f"{slot_name}=n/a")
            continue
        andar = int(slot_row.iloc[0]["andar"])
        bahar = int(slot_row.iloc[0]["bahar"])
        parts.append(f"{slot_name}={andar}/{bahar}")
    return ", ".join(parts)

def _colored_separator(label: str, glyph: str = "🟦", width: int = 12) -> str:
    band = glyph * width
    return f"{band} {label} {band}"

def _write_text(path: Path, content: str) -> None:
    """Write text content to a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)

def _read_prebuilt_info(prebuilt_dir: Path) -> dict:
    info_path = prebuilt_dir / "build_info.json"
    if not info_path.exists():
        return {}
    try:
        return json.loads(info_path.read_text())
    except json.JSONDecodeError:
        return {}


def _parse_date(value: object) -> dt.date | None:
    if isinstance(value, dt.date):
        return value
    if not value:
        return None
    try:
        return dt.date.fromisoformat(str(value))
    except ValueError:
        return None


def _resolve_rebuild_window(
    latest_result_date: dt.date,
    aligned_results: List[dt.date],
    coverage_start: dt.date | None,
    coverage_end: dt.date | None,
    rebuild_choice: str,
    start_raw: str = "",
    end_raw: str = "",
    default_window_days: int = DEFAULT_WINDOW_DAYS,
) -> tuple[dt.date, dt.date]:
    window_fallback_start = latest_result_date - dt.timedelta(days=max(default_window_days - 1, 0))
    if aligned_results:
        window_fallback_start = max(window_fallback_start, min(aligned_results))

    if rebuild_choice == "Y":
        start_date = _prompt_date(start_raw, window_fallback_start) or window_fallback_start
        end_date = _prompt_date(end_raw, latest_result_date) or latest_result_date
    else:
        if coverage_start and coverage_end:
            start_date = coverage_start
            end_date = coverage_end
        else:
            start_date = window_fallback_start
            end_date = latest_result_date

    if start_date > end_date:
        start_date, end_date = end_date, start_date
    return start_date, end_date


def _handle_prebuilt_metrics(
    latest_result_date: dt.date, aligned_results: List[dt.date], cfg: PnLConfig
) -> tuple[dict[str, pd.DataFrame], List[str], dt.date, dt.date]:
    notes: List[str] = []
    info = _read_prebuilt_info(PREBUILT_DIR)
    needs_rebuild, reason = prebuilt_metrics_status(latest_result_date, PREBUILT_DIR)
    start_raw_info = info.get("start") if isinstance(info, dict) else None
    end_raw_info = info.get("end") if isinstance(info, dict) else None
    coverage_start = _parse_date(start_raw_info)
    coverage_end = _parse_date(end_raw_info)

    freshness = "🟥 [STALE]" if needs_rebuild else "🟩 [FRESH]"
    if start_raw_info and end_raw_info:
        print(f"Prebuilt metrics: {freshness} [{start_raw_info} → {end_raw_info}]")
    else:
        print(f"Prebuilt metrics: {freshness}")

    if needs_rebuild:
        notes.append(f"Metrics: {reason}")

    choice = input("Rebuild metrics? (Y/N) [Y]: ").strip().upper()
    if not choice:
        choice = "Y"

    start_raw = ""
    end_raw = ""
    if choice == "Y":
        start_raw = input("Start date? (DD-mm-yy): ").strip()
        end_raw = input("End date? (blank = results file last date): ").strip()

    window_start, window_end = _resolve_rebuild_window(
        latest_result_date,
        aligned_results,
        coverage_start,
        coverage_end,
        choice,
        start_raw=start_raw,
        end_raw=end_raw,
    )

    if choice == "Y":
        metrics = rebuild_prebuilt_metrics(window_start, window_end, cfg)
        if metrics:
            span_days = (window_end - window_start).days + 1
            print(f"[REBUILT] Rebuilt metrics for {window_start}→{window_end} ({span_days} days)")
            return metrics, notes, window_start, window_end

    metrics = load_prebuilt_metrics(PREBUILT_DIR)
    if not metrics:
        notes.append("Metrics: No prebuilt data available")
        print("[WARNING] No prebuilt metrics found")
    else:
        span_days = (window_end - window_start).days + 1
        print(f"📊 Using existing metrics [{window_start} → {window_end} | {span_days} days]")

    return metrics, notes, window_start, window_end


def _load_prediction_history() -> tuple[pd.DataFrame, bool]:
    if not HISTORY_PATH.exists():
        return pd.DataFrame(), True

    history = pd.read_csv(HISTORY_PATH)
    if "prediction_date" in history.columns and "date" not in history.columns:
        history["date"] = history["prediction_date"]
    if "date" in history.columns:
        history["date"] = pd.to_datetime(history["date"], errors="coerce").dt.date
    if "slot" in history.columns and history["slot"].dtype == object:
        history["slot"] = history["slot"].map(SLOT_NAME_TO_ID).fillna(history["slot"])
    return history, False


def _append_history(history: pd.DataFrame, new_rows: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([history, new_rows], ignore_index=True)
    if not combined.empty:
        combined["date"] = pd.to_datetime(combined["date"], errors="coerce").dt.date
        combined = combined.dropna(subset=["date", "slot", "number"])
        combined["prediction_date"] = pd.to_datetime(combined["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        combined = combined.drop_duplicates(subset=["date", "slot", "number"], keep="last")
    combined.to_csv(HISTORY_PATH, index=False)
    return combined


def _collapse_date_gaps(
    effective_dates: List[dt.date], present_dates: List[dt.date], reason_detail: str | None = None
) -> List[str]:
    missing = sorted(set(effective_dates) - set(present_dates))
    if not missing:
        return []

    ranges: List[tuple[dt.date, dt.date]] = []
    start = end = missing[0]
    for current in missing[1:]:
        if current <= end + dt.timedelta(days=2):
            end = current
            continue

        ranges.append((start, end))
        start = end = current
    ranges.append((start, end))

    lines: List[str] = []
    reason_suffix = f" ({reason_detail})." if reason_detail else ""

    for start_date, end_date in ranges:
        if start_date == end_date:
            lines.append(f"SKIP: no predictions for {start_date}{reason_suffix}")
        else:
            lines.append(f"SKIP: no predictions for {start_date}→{end_date}{reason_suffix}")
    return lines


def _weight_sort_columns(group: pd.DataFrame) -> Tuple[List[str], List[bool]]:
    cols: List[str] = []
    ascending: List[bool] = []

    for col, asc in (
        ("score", False),
        ("votes", False),
        ("in_top", False),
        ("rank", True),
        ("stake", False),
    ):
        if col in group.columns:
            cols.append(col)
            ascending.append(asc)

    cols.append("number")
    ascending.append(True)

    return cols, ascending


def _format_pnl_table(
    report_df: pd.DataFrame,
    hit_notes: pd.DataFrame,
    actual_map: Dict[int, int | None],
    latest_date: dt.date,
    gate_map: Dict[str, bool],
    candidates: Dict[str, int],
) -> str:
    if report_df.empty and not actual_map:
        return "No P&L data available"

    latest_day_df = report_df[report_df["date"] == latest_date]
    slots = sorted(set(latest_day_df["slot"].unique()).union(actual_map.keys()))

    prev_day_andars: Dict[int, int] = {}
    prev_day_bahars: Dict[int, int] = {}

    if HISTORY_PATH.exists():
        history_df = pd.read_csv(HISTORY_PATH)
        if "prediction_date" in history_df.columns and "date" not in history_df.columns:
            history_df["date"] = history_df["prediction_date"]
        if "date" in history_df.columns:
            history_df["date"] = pd.to_datetime(history_df["date"], errors="coerce").dt.date
        if "slot" in history_df.columns and history_df["slot"].dtype == object:
            history_df["slot"] = history_df["slot"].map(SLOT_NAME_TO_ID).fillna(history_df["slot"])
        prev_day_df = history_df[history_df["date"] == latest_date]

        if not prev_day_df.empty and "slot" in prev_day_df.columns:
            for slot_id in SLOT_NAME_MAP.keys():
                slot_group = prev_day_df[prev_day_df["slot"] == slot_id]
                if slot_group.empty:
                    continue
                sort_cols, ascending = _weight_sort_columns(slot_group)
                heaviest = slot_group.sort_values(by=sort_cols, ascending=ascending).iloc[0]
                prev_day_andars[slot_id] = int(heaviest["number"]) // 10
                prev_day_bahars[slot_id] = int(heaviest["number"]) % 10

    lines = ["💰 YESTERDAY'S P&L ({})".format(latest_date.strftime("%d-%m-%y"))]
    lines.append("Slot    Result  Picks  Stake  Return   P&L     ROI      AB    AB P&L")
    lines.append("----    ------  -----  -----  ------   ----    ---      --    ------")

    total_picks = 0
    total_stake = 0.0
    total_return = 0.0
    total_pnl = 0.0
    total_ab_pnl = 0.0

    for slot_id in slots:
        slot_df = latest_day_df[latest_day_df["slot"] == slot_id]
        slot_name = SLOT_NAME_MAP.get(slot_id, str(slot_id))
        actual_val = actual_map.get(slot_id)
        actual_display = "XX" if actual_val is None else f"{int(actual_val):02d}"

        picks = len(slot_df)
        total_picks += picks
        stake = slot_df["cost"].sum() if not slot_df.empty else 0.0
        returns = slot_df["payout"].sum() if not slot_df.empty else 0.0
        pnl = slot_df["pnl"].sum() if not slot_df.empty else 0.0
        roi = (pnl / stake * 100) if stake else 0.0

        total_stake += stake
        total_return += returns
        total_pnl += pnl

        hit_rows = slot_df[slot_df["hit"]]
        result_symbol = "[HIT]" if not hit_rows.empty else "[MISS]"

        ab_status = "-"
        ab_pnl_val = 0.0

        if gate_map.get(slot_name, False) and actual_val is not None:
            actual_andar = actual_val // 10
            actual_bahar = actual_val % 10

            prev_andar = prev_day_andars.get(slot_id)
            prev_bahar = prev_day_bahars.get(slot_id)

            if prev_andar is not None and prev_bahar is not None:
                andar_hit = actual_andar == prev_andar
                bahar_hit = actual_bahar == prev_bahar

                if andar_hit and bahar_hit:
                    ab_status = "AB"
                elif andar_hit:
                    ab_status = "A"
                elif bahar_hit:
                    ab_status = "B"

                ab_return = (90 if andar_hit else 0) + (90 if bahar_hit else 0)
                ab_pnl_val = ab_return - 20
        elif gate_map.get(slot_name, False):
            ab_status = "-"

        total_ab_pnl += ab_pnl_val

        ab_pnl_display = f"{ab_pnl_val:>+4.0f}"

        lines.append(
            f"{slot_name:<6}  {actual_display:<6}  {picks:>5}  {stake:>5.0f}  "
            f"{returns:>6.0f}  {pnl:>+5.0f}  {roi:>+5.0f}%  {ab_status:>4}  "
            f"{ab_pnl_display}  {result_symbol}"
        )

    lines.append("-" * 70)
    total_roi = (total_pnl / total_stake * 100) if total_stake else 0.0
    lines.append(
        f"Total:           {total_picks:>5}  {total_stake:>5.0f}  {total_return:>6.0f}  "
        f"{total_pnl:>+5.0f}  {total_roi:>+5.0f}%         {total_ab_pnl:>+4.0f}"
    )

    return "\n".join(lines)


def _format_rollups(combined_totals: pd.DataFrame) -> List[str]:
    lines: List[str] = []

    def _signal(roi: float) -> str:
        if roi > 0.15:
            return "[GOOD]"
        if roi > 0:
            return "[OK]"
        return "[BAD]"

    for name in ("day", "7d", "month", "cumulative"):
        row = combined_totals[combined_totals["window"] == name]
        if row.empty:
            continue

        r = row.iloc[0]
        roi_display = f"{r['roi']:+.1%}"

        stake_display = f"{r['stake']/1000:.1f}K" if r['stake'] >= 1000 else f"{r['stake']:.0f}"

        signal = _signal(r["roi"])
        lines.append(
            f"{name.title():<11} {signal:<7} Stake: {stake_display:<6} P&L: {r['pnl']:>+7.0f} ROI: {roi_display}"
        )

    return lines


def _stake_summary(cfg: PnLConfig) -> str:
    return f"Unit: ₹{cfg.cost_per_unit:.0f}/pick | Main: {cfg.payout_per_unit:.0f}x | Digit: {cfg.digit_payout_per_unit:.0f}x"


def _render_daily_report(
    pnl_table: str,
    day_topk_lines: List[str],
    rank_lines: List[str],
    tag_lines: List[str],
    cross_slot_lines: List[str],
    ab_gate_lines: List[str],
    rollups: List[str],
    hero_lines: List[str],
    stake_line: str,
    trimmed_notes: List[str],
    skip_lines: List[str],
    notes: List[str],
) -> str:
    parts: List[str] = []

    parts.append(_colored_separator("DAILY REPORT", glyph="🟪"))
    parts.append(stake_line)
    parts.append("")

    if skip_lines:
        parts.extend(skip_lines)
        parts.append("")

    parts.append(pnl_table)
    parts.append("")

    if day_topk_lines:
        parts.append(day_topk_lines[0])
        parts.append("")

    if ab_gate_lines:
        parts.extend(ab_gate_lines)
        parts.append("")

    if rank_lines:
        parts.extend(rank_lines)
        parts.append("")

    if tag_lines:
        parts.extend(tag_lines)
        parts.append("")

    if cross_slot_lines:
        parts.extend(cross_slot_lines)
        parts.append("")

    if hero_lines:
        parts.extend(hero_lines)
        parts.append("")

    if rollups:
        parts.append(_colored_separator("ROI BANNERS", glyph="🟩"))
        parts.append("Period      Status  Stake     P&L        ROI")
        parts.append("-------     ------  -----     ----       ---")
        parts.extend(rollups)
        parts.append("")

    if trimmed_notes:
        parts.append("K-AUTO NOTES")
        for note in trimmed_notes:
            parts.append(f"• {note}")
        parts.append("")

    if notes:
        parts.append("📝 NOTES")
        for note in notes:
            parts.append(f"• {note}")

    return "\n".join(parts)

def _validate_shortlist(shortlist: pd.DataFrame, k_auto_map: Dict[int, int]) -> List[str]:
    """Validate shortlist against K-AUTO limits and return warnings."""
    warnings = []
    
    if shortlist.empty:
        warnings.append("⚠️  Empty shortlist - no predictions generated")
        return warnings
    
    # Check if shortlist has required columns
    required_cols = ["slot", "rank", "number", "score"]
    missing_cols = [col for col in required_cols if col not in shortlist.columns]
    if missing_cols:
        warnings.append(f"⚠️  Missing columns in shortlist: {', '.join(missing_cols)}")
    
    # Check picks per slot
    for slot_name in SLOT_NAME_TO_ID.keys():
        slot_id = SLOT_NAME_TO_ID[slot_name]
        slot_data = shortlist[shortlist["slot"] == slot_name]
        
        if slot_data.empty:
            warnings.append(f"⚠️  No predictions for slot {slot_name}")
            continue
        
        num_picks = len(slot_data)
        expected_cap = k_auto_map.get(slot_id, MAX_PICKS_CAP_DEFAULT)
        
        if num_picks > expected_cap:
            warnings.append(f"⚠️  {slot_name} has {num_picks} picks (expected max {expected_cap})")
        elif num_picks < 5:
            warnings.append(f"⚠️  {slot_name} has only {num_picks} picks (unusually low)")
    
    return warnings

def _save_shortlist_with_history(shortlist: pd.DataFrame, prediction_date: dt.date) -> None:
    """Save shortlist with historical tracking."""
    # Save to date-specific folder
    date_folder = OUTPUT_DIR / prediction_date.strftime("%Y-%m-%d")
    date_folder.mkdir(parents=True, exist_ok=True)
    
    # Save CSV and Excel
    shortlist.to_csv(date_folder / "shortlist.csv", index=False)
    shortlist.to_excel(date_folder / "shortlist.xlsx", index=False)
    
    # Append to history file
    shortlist_copy = shortlist.copy()
    shortlist_copy["prediction_date"] = prediction_date.strftime("%Y-%m-%d")
    
    if HISTORY_PATH.exists():
        history_df = pd.read_csv(HISTORY_PATH)
        history_df = pd.concat([history_df, shortlist_copy], ignore_index=True)
    else:
        history_df = shortlist_copy
    
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    history_df.to_csv(HISTORY_PATH, index=False)


def _write_metadata(date_folder: Path, metadata: Dict[str, object]) -> None:
    date_folder.mkdir(parents=True, exist_ok=True)
    metadata_path = date_folder / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)


def _compute_config_hash(payload: Dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()


def generate_for_single_day(
    prediction_date: dt.date,
    cutoff_date: dt.date,
    k_auto_map: Dict[int, int],
    project_root: Path,
    mode: str = "historic",
    gate_cutoff_date: Optional[dt.date] = None,
    scr_timeout: int = 300,
    scr_retries: int = 1,
) -> Optional[pd.DataFrame]:
    """Generate predictions for a single day with an optional cutoff date."""
    _run_scripts_quietly(
        project_root,
        cutoff_date=cutoff_date,
        scr_timeout=scr_timeout,
        scr_retries=scr_retries,
        scr2_date=prediction_date,
    )
    try:
        shortlist = _load_shortlist()
    except Exception as e:
        logger.warning(f"  SKIP: Could not load generated shortlist: {e}")
        return None

    if shortlist.empty:
        logger.warning(f"  SKIP: Generated shortlist is empty for {_format_ddmmyy(prediction_date)}")
        return None

    shortlist, trim_notes = _apply_max_cap(shortlist, k_auto_map)
    _save_shortlist_with_history(shortlist, prediction_date)
    gate_cutoff = gate_cutoff_date or cutoff_date
    gate_snapshot = compute_ab_gate_snapshot(gate_cutoff)
    _generate_daily_report(
        prediction_date,
        shortlist,
        trim_notes,
        k_auto_map,
        gate_snapshot=gate_snapshot,
        cutoff_date=gate_cutoff,
    )
    if logger.isEnabledFor(logging.DEBUG):
        logger.info(f"  [auto-generated] {mode} predictions saved for {_format_ddmmyy(prediction_date)}")
    return shortlist


def _load_or_generate_shortlist(
    target_date: dt.date,
    k_auto_map: Dict[int, int],
    project_root: Path,
    auto_generate: bool,
    ab_cutoff: str,
    scr_timeout: int = 300,
    scr_retries: int = 1,
) -> Optional[pd.DataFrame]:
    """Load saved shortlist for a date, or generate + save if missing."""
    try:
        shortlist = _load_shortlist_for_date(target_date)
        if shortlist.empty:
            return shortlist
        if logger.isEnabledFor(logging.DEBUG):
            logger.info(f"  [saved] Using saved predictions for {_format_ddmmyy(target_date)}")
        return shortlist
    except FileNotFoundError:
        if not auto_generate:
            logger.warning(
                f"  SKIP: No saved predictions for {target_date.strftime('%d-%m-%y')} "
                "(auto-generate disabled)."
            )
            return None
        logger.info(
            f"  Missing predictions for {target_date.strftime('%d-%m-%y')} - auto-generating..."
        )
    except Exception as e:
        logger.warning(f"  SKIP: Error loading predictions: {e}")
        return None

    shortlist = generate_for_single_day(
        target_date,
        cutoff_date=target_date - dt.timedelta(days=1),
        gate_cutoff_date=_resolve_ab_cutoff_date(target_date, ab_cutoff),
        k_auto_map=k_auto_map,
        project_root=project_root,
        mode="historic",
        scr_timeout=scr_timeout,
        scr_retries=scr_retries,
    )
    if shortlist is not None:
        logger.info("  ✅ Predictions generated and saved.")
    return shortlist


def sanity_tests(
    auto_generate_missing: bool,
    ab_cutoff: str,
    scr_timeout: int,
    scr_retries: int,
) -> None:
    """Run lightweight sanity checks in verbose mode only."""
    if not logger.isEnabledFor(logging.DEBUG):
        return

    test_date = dt.date(2025, 7, 17)
    project_root = Path(__file__).resolve().parent

    try:
        shortlist = _load_shortlist_for_date(test_date)
    except FileNotFoundError:
        if not auto_generate_missing:
            logger.info("[SELF-TEST SKIP] Saved shortlist missing; auto-generation disabled.")
            return
        logger.info("[SELF-TEST] Saved shortlist missing; attempting auto-generation with cutoff.")
        default_map = {slot_id: MAX_PICKS_CAP_DEFAULT for slot_id in SLOT_NAME_MAP.keys()}
        try:
            generate_for_single_day(
                test_date,
                cutoff_date=test_date - dt.timedelta(days=1),
                gate_cutoff_date=_resolve_ab_cutoff_date(test_date, ab_cutoff),
                k_auto_map=default_map,
                project_root=project_root,
                mode="historic",
                scr_timeout=scr_timeout,
                scr_retries=scr_retries,
            )
            logger.info("[SELF-TEST PASS] auto-generation completed without crash.")
        except Exception as exc:
            logger.warning(f"[SELF-TEST FAIL] auto-generation crashed: {exc}")
        return

    if shortlist.empty:
        logger.warning("[SELF-TEST SKIP] Saved shortlist is empty.")
        return

    preds = _prepare_predictions(shortlist, test_date)
    if preds.empty:
        logger.warning("[SELF-TEST SKIP] No predictions available after prep.")
        return

    cfg = PnLConfig()
    gate_by_day = {
        test_date: compute_ab_gate_snapshot(_resolve_ab_cutoff_date(test_date, ab_cutoff))
    }
    report = compute_pnl_report(preds, cfg, gate_by_day=gate_by_day)
    merged = report.merged
    digit_pnl = report.digit_pnl

    if merged.empty:
        logger.warning("[SELF-TEST SKIP] No merged results for test date.")
        return

    main_day = merged[merged["date"] == test_date][["cost", "payout", "pnl"]].sum()
    ab_day = digit_pnl[digit_pnl["date"] == test_date][["cost", "payout", "pnl"]].sum()

    if main_day.empty:
        logger.warning("[SELF-TEST SKIP] Test date missing from P&L output.")
        return

    main_pnl = float(main_day.get("pnl", 0.0))
    ab_pnl = float(ab_day.get("pnl", 0.0))
    total_pnl = main_pnl + ab_pnl

    expected_ab = 70 if ab_cutoff == "same" else ab_pnl
    expected_total = 170 if ab_cutoff == "same" else total_pnl
    if round(main_pnl) == 100 and round(ab_pnl) == round(expected_ab) and round(total_pnl) == round(expected_total):
        logger.info(
            "[SELF-TEST PASS] 17-07-25 (main %+0.0f, AB %+0.0f, total %+0.0f).",
            main_pnl,
            ab_pnl,
            total_pnl,
        )
    else:
        logger.warning(
            f"[SELF-TEST FAIL] 17-07-25 P&L mismatch: main {main_pnl:+.0f}, "
            f"AB {ab_pnl:+.0f}, total {total_pnl:+.0f}."
        )


def run_interactive_display(
    scr_timeout: int = 300,
    scr_retries: int = 1,
) -> None:
    notes: List[str] = []
    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.simplefilter("always", DeprecationWarning)

        project_root = Path(__file__).resolve().parent
        results_df = load_results_dataframe()
        result_dates = pd.to_datetime(results_df["DATE"], errors="coerce").dropna().dt.date.tolist()
        aligned_results = [d for d in result_dates if not _is_month_end(d)]
        if not aligned_results:
            print("❌ SKIP: No usable result dates available")
            return

        latest_date = _latest_non_month_end_date(result_dates)
        prediction_date = _next_non_month_end(latest_date + dt.timedelta(days=1))

        cfg = PnLConfig()
        prebuilt_metrics, prebuilt_notes, window_start, window_end = _handle_prebuilt_metrics(
            latest_date, aligned_results, cfg
        )

        print(f"\n📅 Generating predictions for {prediction_date:%d-%m-%y}...")
        print(_colored_separator("RUNNING SCR SCRIPTS", glyph="🟦"))

        _run_scripts_quietly(
            project_root,
            cutoff_date=latest_date,
            scr_timeout=scr_timeout,
            scr_retries=scr_retries,
            scr2_date=prediction_date,
        )

        effective_dates = build_effective_dates(window_start, window_end, available_dates=result_dates)

        bet_rows = load_clean_bet_rows(window_start, window_end, cfg)
        rank_lines, k_auto_map = format_rank_bucket_windows(bet_rows, effective_dates)
        _, day_topk_lines = format_topk_profit(
            bet_rows, effective_dates=effective_dates, unit_stake=cfg.cost_per_unit
        )
        tag_lines = format_tag_roi(bet_rows, effective_dates=effective_dates, unit_stake=cfg.cost_per_unit)
        cross_slot_lines = format_cross_slot_hits(bet_rows, effective_dates=effective_dates)

        prediction_files_missing = False
        try:
            shortlist = _load_shortlist()
        except FileNotFoundError:
            prediction_files_missing = True
            shortlist = pd.DataFrame(
                columns=["slot", "number", "rank", "score", "votes", "sources", "tier", "in_top"]
            )

        shortlist, trimmed_notes = _apply_max_cap(shortlist, k_auto_map)
        if not prediction_files_missing and not shortlist.empty:
            _save_shortlist_with_history(shortlist, prediction_date)

        candidates = _strongest_candidates(shortlist)
        candidates_map = {name: num for name, num in candidates}

        print("\n✅ STRONGEST ANDAR/BAHAR DIGITS")
        for slot_name, number in candidates:
            tens = number // 10
            ones = number % 10
            print(f"{slot_name} → {number:02d} (tens:{tens}, ones:{ones})")

        print("\n📊 FINAL BET NUMBERS")
        bet_lines = _slot_bet_lines(shortlist, trimmed_notes)
        for line in bet_lines:
            print(line)

        preds = _prepare_predictions(shortlist, prediction_date)
        history, missing_history_file = _load_prediction_history()
        prediction_files_missing = prediction_files_missing or missing_history_file
        history = _append_history(history, preds)

        history_for_results = history[history["date"] <= latest_date]
        report = compute_pnl_report(history_for_results, cfg=cfg)
        actual_map: Dict[int, int | None] = {}
        latest_results = results_df[pd.to_datetime(results_df["DATE"]).dt.date == latest_date]
        if not latest_results.empty:
            for slot_name, value in latest_results.iloc[0].items():
                if slot_name == "DATE":
                    continue
                slot_id = SLOT_NAME_TO_ID.get(slot_name)
                if slot_id is None:
                    continue
                try:
                    actual_map[slot_id] = int(value)
                except (TypeError, ValueError):
                    actual_map[slot_id] = None

        ab_gate_lines, gate_map = format_andar_bahar_gating(
            report.slot_digit_hits, cfg, effective_dates
        )

        pnl_table = _format_pnl_table(
            report.merged,
            report.hit_notes,
            actual_map,
            latest_date,
            gate_map,
            candidates_map,
        )
        rollups = _format_rollups(report.combined_window_totals)
        notes.extend(prebuilt_notes)

        hero_lines: List[str] = []
        if prebuilt_metrics:
            hero_lines = format_hero_weakest(prebuilt_metrics, min_bets=20)

        stake_line = _stake_summary(cfg)

        present_dates = history_for_results["date"].dropna().tolist()
        skip_reason = "missing prediction files" if prediction_files_missing else "no predictions logged"
        skip_range_lines = _collapse_date_gaps(effective_dates, present_dates, skip_reason)

        day_topk_lines = [line for line in day_topk_lines if line]
        rank_lines = [line for line in rank_lines if line]
        tag_lines = [line for line in tag_lines if line]
        cross_slot_lines = [line for line in cross_slot_lines if line]
        hero_lines = [line for line in hero_lines if line]
        ab_gate_lines = [line for line in ab_gate_lines if line]

    for warning in captured_warnings:
        if issubclass(warning.category, DeprecationWarning):
            notes.append(f"DeprecationWarning: {warning.message}")

    daily_report_body = _render_daily_report(
        pnl_table,
        day_topk_lines,
        rank_lines,
        tag_lines,
        cross_slot_lines,
        ab_gate_lines,
        rollups,
        hero_lines,
        stake_line,
        trimmed_notes,
        skip_range_lines,
        notes,
    )

    pnl_summary_body = render_compact_report(report)

    daily_report_path = Path("reports/daily_report_auto.txt")
    pnl_path = Path("logs/pnl/pnl_summary_auto.txt")
    _write_text(daily_report_path, daily_report_body)
    _write_text(pnl_path, pnl_summary_body)

    print("\n" + _colored_separator("DAILY REPORT", glyph="🟩"))
    print(daily_report_body)
    print(f"\n✅ Daily report saved to: {daily_report_path}")


def run_backtest(
    num_days: int,
    rebuild_metrics: bool = False,
    auto_generate_missing: bool = True,
    ab_cutoff: str = "same",
    scr_timeout: int = 300,
    scr_retries: int = 1,
) -> None:
    """Run backtest for the last N days from results file."""
    needs_rebuild, reason = _metrics_needs_rebuild()
    should_rebuild = bool(rebuild_metrics and needs_rebuild)

    logger.info(f"\n{'='*60}")
    logger.info(f"BACKTEST MODE: Last {num_days} Days")
    if should_rebuild:
        logger.info("WITH METRICS REBUILD")
    else:
        logger.info("NO METRICS REBUILD")
    logger.info(f"Auto-generate missing predictions: {'ON' if auto_generate_missing else 'OFF'}")
    logger.info(f"AB cutoff: {ab_cutoff}")
    logger.info(f"{'='*60}")
    
    # Load results data
    logger.debug("Loading results dataframe...")
    results_df = load_results_dataframe()
    results_df["DATE"] = pd.to_datetime(results_df["DATE"], errors="coerce").dt.date
    
    # Filter out month-end dates
    results_df = results_df[~results_df["DATE"].apply(_is_month_end)]
    
    # Filter out rows with NaN in slot columns
    slot_columns = list(SLOT_NAME_TO_ID.keys())
    results_df = results_df.dropna(subset=slot_columns)
    
    if results_df.empty:
        logger.error("ERROR: No usable result dates available")
        return
    
    # Get last N days
    available_dates = sorted(results_df["DATE"].unique(), reverse=True)
    if len(available_dates) < num_days:
        logger.warning(f"WARNING: Only {len(available_dates)} days available, testing all of them")
        num_days = len(available_dates)
    
    backtest_dates = available_dates[:num_days]
    start_date = min(backtest_dates)
    end_date = max(backtest_dates)
    
    # Rebuild metrics if requested
    if should_rebuild:
        logger.info("\n🔄 Rebuilding metrics from historical data...")
        _rebuild_metrics_if_needed()
    elif rebuild_metrics:
        logger.debug("Metrics rebuild skipped: %s", reason)
    
    logger.info("-" * 60)
    
    # Run the backtest using the date range function
    run_backtest_date_range(
        start_date,
        end_date,
        auto_generate_missing=auto_generate_missing,
        ab_cutoff=ab_cutoff,
        scr_timeout=scr_timeout,
        scr_retries=scr_retries,
    )

def main() -> None:
    """Main entry point for the console runner."""
    parser = argparse.ArgumentParser(
        description="Precise Predictor Console Runner - Backtest & Future Prediction Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Backtest commands
  py -3.12 run_minimal_console.py --backtest --last 30
  py -3.12 run_minimal_console.py --backtest --last 30 --rebuild-metrics
  py -3.12 run_minimal_console.py --backtest --start 01-07-25 --end 15-07-25
  py -3.12 run_minimal_console.py --backtest --start 01-07-25 --end 15-07-25 --rebuild-metrics
  py -3.12 run_minimal_console.py --backtest  # Interactive mode
  
  # Generate future predictions
  py -3.12 run_minimal_console.py --generate-future  # Auto-detect, interactive
  py -3.12 run_minimal_console.py --generate-future --rebuild-metrics  # Auto-detect with rebuild
  py -3.12 run_minimal_console.py --generate-future --from 01-04-25 --to 20-12-25  # Custom range
  py -3.12 run_minimal_console.py --generate-future --until 19-07-25  # Auto-detect start, generate up to date
  py -3.12 run_minimal_console.py --check-git  # Print git status and exit
        """
    )
    
    # Backtest arguments
    parser.add_argument("--backtest", action="store_true", 
                       help="Run in backtest mode")
    parser.add_argument(
        "--last",
        type=int,
        help="Number of days to backtest (use with --backtest). Common: 30/60/90/120/150/180/365",
    )
    parser.add_argument("--start", type=str, 
                       help="Start date for backtest (DD-MM-YY format)")
    parser.add_argument("--end", type=str, 
                       help="End date for backtest (DD-MM-YY format)")
    parser.add_argument(
        "--auto-generate-missing",
        dest="auto_generate_missing",
        action="store_true",
        default=True,
        help="Auto-generate missing predictions during backtest (default: ON)",
    )
    parser.add_argument(
        "--no-auto-generate-missing",
        dest="auto_generate_missing",
        action="store_false",
        help="Disable auto-generation of missing predictions during backtest",
    )
    
    # Future prediction arguments
    parser.add_argument("--generate-future", action="store_true", 
                       help="Generate future predictions")
    parser.add_argument("--from", dest="from_date", type=str, 
                       help="Start date for future generation (DD-MM-YY format)")
    parser.add_argument("--to", "--until", dest="to_date", type=str, 
                       help="End date for future generation (DD-MM-YY format). "
                            "Can be used alone (auto-detects start from last prediction or latest Excel data) "
                            "or with --from for explicit date range")
    
    # Common arguments
    parser.add_argument("--rebuild-metrics", action="store_true", 
                       help="Rebuild metrics from historical data before processing")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging for debugging")
    parser.add_argument("--check-git", action="store_true",
                       help="Print git status summary and exit")
    parser.add_argument("--no-push", action="store_true",
                       help="Skip git status/push operations")
    parser.add_argument("--dry-run", action="store_true",
                       help="Generate future ranges without writing files")
    parser.add_argument(
        "--ab-cutoff",
        choices=("prev", "same"),
        default="same",
        help="AB snapshot cutoff: prev=D-1 or same=D (default: same)",
    )
    parser.add_argument(
        "--scr-timeout",
        type=int,
        default=300,
        help="Timeout in seconds for SCR2 execution (default: 300)",
    )
    parser.add_argument(
        "--scr-retries",
        type=int,
        default=1,
        help="Retries for SCR2 on non-zero exit or timeout (default: 1)",
    )
    
    args = parser.parse_args()

    global AUTO_PUSH_ENABLED
    AUTO_PUSH_ENABLED = not args.no_push
    
    # Set logging level based on verbose flag
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")
    
    # Track if script executed successfully
    success = False
    
    if args.check_git and not (args.backtest or args.generate_future):
        _check_git_status(explicit=True)
        return

    if args.check_git:
        _check_git_status(explicit=True)

    # Handle backtest mode
    if args.backtest:
        try:
            handle_backtest_mode(args)
            success = True
        except SystemExit:
            # Allow sys.exit() to propagate
            raise
        except Exception as e:
            logger.error(f"ERROR: Backtest failed: {e}")
            sys.exit(1)
    # Handle future generation mode
    elif args.generate_future:
        try:
            handle_future_generation_mode(args)
            success = True
        except SystemExit:
            # Allow sys.exit() to propagate
            raise
        except Exception as e:
            logger.error(f"ERROR: Future generation failed: {e}")
            sys.exit(1)
    else:
        run_interactive_display(
            scr_timeout=args.scr_timeout,
            scr_retries=args.scr_retries,
        )
        success = True

def handle_backtest_mode(args) -> None:
    """Handle all backtest command variations."""
    sanity_tests(args.auto_generate_missing, args.ab_cutoff, args.scr_timeout, args.scr_retries)
    # Interactive mode - no args provided
    if args.last is None and args.start is None and args.end is None:
        logger.info(f"\n{'='*60}")
        logger.info("BACKTEST - INTERACTIVE MODE")
        logger.info(f"{'='*60}\n")
        
        logger.info("Select backtest option:")
        logger.info("  [1] Last N days")
        logger.info("  [2] Custom date range (DD-MM-YY)")
        logger.info("  [3] Cancel")
        
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == "1":
            try:
                num_days = int(input("Enter number of days to backtest: ").strip())
                rebuild = input("Rebuild metrics? (y/n): ").strip().lower() == 'y'
                run_backtest(
                    num_days,
                    rebuild_metrics=rebuild,
                    auto_generate_missing=args.auto_generate_missing,
                    ab_cutoff=args.ab_cutoff,
                    scr_timeout=args.scr_timeout,
                    scr_retries=args.scr_retries,
                )
            except ValueError:
                logger.error("ERROR: Invalid number of days")
                sys.exit(1)
        elif choice == "2":
            try:
                start_str = input("Enter start date (DD-MM-YY): ").strip()
                end_str = input("Enter end date (DD-MM-YY): ").strip()
                start_date, end_date = parse_date_range(start_str, end_str)
                rebuild = input("Rebuild metrics? (y/n): ").strip().lower() == 'y'
                
                if rebuild:
                    _maybe_rebuild_metrics(log_skip=True)
                
                run_backtest_date_range(
                    start_date,
                    end_date,
                    auto_generate_missing=args.auto_generate_missing,
                    ab_cutoff=args.ab_cutoff,
                    scr_timeout=args.scr_timeout,
                    scr_retries=args.scr_retries,
                )
            except ValueError as e:
                logger.error(f"ERROR: {e}")
                sys.exit(1)
        elif choice == "3":
            logger.info("Cancelled")
            sys.exit(0)
        else:
            logger.error("ERROR: Invalid choice")
            sys.exit(1)
        return
    
    # Date range mode
    if args.start and args.end:
        try:
            start_date, end_date = parse_date_range(args.start, args.end)
            
            if args.rebuild_metrics:
                _maybe_rebuild_metrics(log_skip=True)
            
            run_backtest_date_range(
                start_date,
                end_date,
                auto_generate_missing=args.auto_generate_missing,
                ab_cutoff=args.ab_cutoff,
                scr_timeout=args.scr_timeout,
                scr_retries=args.scr_retries,
            )
        except ValueError as e:
            logger.error(f"ERROR: {e}")
            sys.exit(1)
        return
    
    # Last N days mode
    if args.last is not None:
        if args.last <= 0:
            logger.error("ERROR: --last must be a positive integer")
            sys.exit(1)
        run_backtest(
            args.last,
            rebuild_metrics=args.rebuild_metrics,
            auto_generate_missing=args.auto_generate_missing,
            ab_cutoff=args.ab_cutoff,
            scr_timeout=args.scr_timeout,
            scr_retries=args.scr_retries,
        )
        return
    
    # Invalid combination
    logger.error("ERROR: --backtest requires either --last N or --start and --end")
    logger.info("Usage examples:")
    logger.info("  python run_minimal_console.py --backtest --last 30")
    logger.info("  python run_minimal_console.py --backtest --start 01-06-25 --end 18-07-25")
    logger.info("  python run_minimal_console.py --backtest  # Interactive mode")
    sys.exit(1)

def handle_future_generation_mode(args) -> None:
    """Handle all future generation command variations."""
    # Custom date range mode (both --from and --to provided)
    if args.from_date and args.to_date:
        try:
            start_date, end_date = parse_date_range(args.from_date, args.to_date)

            if args.rebuild_metrics:
                _maybe_rebuild_metrics(log_skip=True)

            # Generate predictions for the entire range
            generate_future_predictions_range(
                start_date,
                end_date,
                dry_run=args.dry_run,
                ab_cutoff=args.ab_cutoff,
                scr_timeout=args.scr_timeout,
                scr_retries=args.scr_retries,
            )
        except ValueError as e:
            logger.error(f"ERROR: {e}")
            sys.exit(1)
        return
    
    # --until only mode (backward compatibility): auto-detect start date, use specified end date
    if args.to_date and not args.from_date:
        try:
            # Parse the target end date
            end_date = parse_ddmmyy(args.to_date)

            last_prediction_date = get_last_prediction_date()
            last_results_date = _get_latest_result_date()
            logger.info(
                "Auto start inputs: last_prediction=%s, last_results=%s",
                _format_ddmmyy(last_prediction_date) if last_prediction_date else "n/a",
                _format_ddmmyy(last_results_date) if last_results_date else "n/a",
            )

            start_candidates = []
            if _is_valid_date(last_prediction_date):
                start_candidates.append(last_prediction_date + dt.timedelta(days=1))
            if _is_valid_date(last_results_date):
                start_candidates.append(last_results_date + dt.timedelta(days=1))
            if not start_candidates:
                logger.error("Bad date inference")
                return

            start_date = max(start_candidates)

            if not _is_valid_date(start_date) or not _is_valid_date(end_date):
                logger.error("Bad date inference")
                return

            logger.info(
                "Resolved auto range: start=%s, end=%s",
                _format_ddmmyy(start_date),
                _format_ddmmyy(end_date),
            )

            if end_date < start_date:
                logger.info("Already up to date; nothing to generate.")
                return

            logger.info(
                f"📅 Generating predictions from {_format_ddmmyy(start_date)} to {_format_ddmmyy(end_date)}"
            )

            if args.rebuild_metrics:
                _maybe_rebuild_metrics(log_skip=True)

            # Generate predictions for the range
            generate_future_predictions_range(
                start_date,
                end_date,
                dry_run=args.dry_run,
                ab_cutoff=args.ab_cutoff,
                scr_timeout=args.scr_timeout,
                scr_retries=args.scr_retries,
            )
        except ValueError as e:
            logger.error(f"ERROR: {e}")
            sys.exit(1)
        return
    
    # Auto-detect mode with optional rebuild (no date arguments)
    if not args.from_date and not args.to_date:
        # Rebuild metrics if requested
        if args.rebuild_metrics:
            _maybe_rebuild_metrics(log_skip=True)
        
        # Interactive menu
        logger.info(f"\n{'='*60}")
        logger.info("GENERATE FUTURE PREDICTIONS - AUTO-DETECT MODE")
        logger.info(f"{'='*60}\n")
        
        try:
            # Auto-detect last dates
            latest_excel_date = _get_latest_result_date()
            
            # Check for last prediction date
            last_prediction_date = get_last_prediction_date()
            
            if not _is_valid_date(latest_excel_date):
                logger.error("Bad date inference")
                sys.exit(1)

            logger.info(f"Latest Excel data: {_format_ddmmyy(latest_excel_date)}")
            if last_prediction_date:
                logger.info(f"Last prediction:   {_format_ddmmyy(last_prediction_date)}")
            else:
                logger.info(f"Last prediction:   None found")
            
            # Calculate gap and next date
            if last_prediction_date and last_prediction_date < latest_excel_date:
                gap_days = (latest_excel_date - last_prediction_date).days
                logger.warning(f"\n⚠️  Gap detected: {gap_days} days between last prediction and latest data")
                logger.info(
                    f"Will backfill: {_format_ddmmyy(last_prediction_date + dt.timedelta(days=1))} "
                    f"to {_format_ddmmyy(latest_excel_date)}"
                )
                
                backfill_start = last_prediction_date + dt.timedelta(days=1)
                backfill_end = latest_excel_date
            else:
                backfill_start = None
                backfill_end = None
            
            # Next day (future prediction)
            next_day = _next_non_month_end(latest_excel_date + dt.timedelta(days=1))
            logger.info(f"Next day (future): {_format_ddmmyy(next_day)}")
            
            logger.info("\nWhat would you like to do?")
            logger.info("  [1] Generate predictions (backfill gap + next day only)")
            logger.info("  [2] Custom date range (manual input)")
            logger.info("  [3] Cancel")
            
            choice = input("\nEnter choice (1-3): ").strip()
            
            if choice == "1":
                # Generate backfill + next day
                if backfill_start and backfill_end:
                    logger.info(f"\n📅 Generating backfill predictions...")
                    generate_future_predictions_range(
                        backfill_start,
                        backfill_end,
                        dry_run=args.dry_run,
                        ab_cutoff=args.ab_cutoff,
                        scr_timeout=args.scr_timeout,
                        scr_retries=args.scr_retries,
                    )
                
                logger.info(f"\n📅 Generating next day prediction...")
                generate_future_predictions_range(
                    next_day,
                    next_day,
                    dry_run=args.dry_run,
                    ab_cutoff=args.ab_cutoff,
                    scr_timeout=args.scr_timeout,
                    scr_retries=args.scr_retries,
                )
                
            elif choice == "2":
                start_str = input("Enter start date (DD-MM-YY): ").strip()
                end_str = input("Enter end date (DD-MM-YY): ").strip()
                start_date, end_date = parse_date_range(start_str, end_str)
                generate_future_predictions_range(
                    start_date,
                    end_date,
                    dry_run=args.dry_run,
                    ab_cutoff=args.ab_cutoff,
                    scr_timeout=args.scr_timeout,
                    scr_retries=args.scr_retries,
                )
                
            elif choice == "3":
                logger.info("Cancelled")
                sys.exit(0)
            else:
                logger.error("ERROR: Invalid choice")
                sys.exit(1)
                
        except Exception as e:
            logger.error(f"ERROR: {e}")
            sys.exit(1)
        
        return
    
    # Should not reach here - all cases handled above
    logger.error("ERROR: Invalid argument combination")
    sys.exit(1)

def get_last_prediction_date() -> Optional[dt.date]:
    """Get the last prediction date from dated prediction folders."""
    date_candidates: List[dt.date] = []
    if OUTPUT_DIR.exists():
        for entry in OUTPUT_DIR.iterdir():
            if not entry.is_dir():
                continue
            try:
                parsed = dt.datetime.strptime(entry.name, "%Y-%m-%d").date()
                date_candidates.append(parsed)
            except ValueError:
                continue
    return max(date_candidates) if date_candidates else None


def _get_latest_result_date() -> Optional[dt.date]:
    results_df = load_results_dataframe()
    if results_df.empty:
        return None
    results_df["DATE"] = pd.to_datetime(results_df["DATE"], errors="coerce").dt.date
    results_df = results_df[~results_df["DATE"].apply(_is_month_end)]
    dates = [d for d in results_df["DATE"].tolist() if _is_valid_date(d)]
    return max(dates) if dates else None

def generate_future_predictions_range(
    start_date: dt.date,
    end_date: dt.date,
    dry_run: bool = False,
    ab_cutoff: str = "same",
    scr_timeout: int = 300,
    scr_retries: int = 1,
) -> None:
    """Generate predictions for a specific date range."""
    if not _is_valid_date(start_date) or not _is_valid_date(end_date):
        logger.error("Bad date inference")
        return
    logger.info(f"\n{'='*60}")
    logger.info(f"FUTURE PREDICTIONS GENERATION")
    logger.info(f"Starting from: {_format_ddmmyy(start_date)}")
    logger.info(f"Ending at: {_format_ddmmyy(end_date)}")
    logger.info(f"{'='*60}")
    
    # Calculate total days
    total_days = (end_date - start_date).days + 1
    if total_days <= 0:
        logger.error("ERROR: End date must be on or after start date")
        return

    if dry_run:
        logger.info("DRY RUN: No files will be written.")
        logger.info(f"Total days to process: {total_days}")
        return
    
    logger.info(f"Total days to process: {total_days}")
    
    # Try to load K-AUTO map from historical metrics
    cfg = PnLConfig()
    # Initialize with default values for all slots
    k_auto_map = {slot_id: MAX_PICKS_CAP_DEFAULT for slot_id in SLOT_NAME_MAP.keys()}
    
    try:
        # Get available historical dates for metrics
        results_df = load_results_dataframe()
        results_df["DATE"] = pd.to_datetime(results_df["DATE"], errors="coerce").dt.date
        results_df = results_df[~results_df["DATE"].apply(_is_month_end)]
        
        if not results_df.empty:
            available_dates = sorted([d for d in results_df["DATE"].dropna().tolist() if d < start_date])
            
            if available_dates:
                # Use last 30 days of history before prediction start date
                history_end = max(available_dates)
                history_start = history_end - dt.timedelta(days=30)
                
                effective_dates = build_effective_dates(history_start, history_end, available_dates=available_dates)
                bet_rows = load_clean_bet_rows(history_start, history_end, cfg)
                
                if not bet_rows.empty:
                    _, loaded_k_auto_map = format_rank_bucket_windows(bet_rows, effective_dates)
                    if loaded_k_auto_map:
                        k_auto_map.update(loaded_k_auto_map)  # Update with loaded values
                        logger.info(f"K-AUTO limits from historical data: {', '.join([f'{SLOT_NAME_MAP[sid]}={cap}' for sid, cap in k_auto_map.items()])}")
        
        if not any(v != MAX_PICKS_CAP_DEFAULT for v in k_auto_map.values()):
            logger.warning(f"⚠️  No historical data available for K-AUTO, using default cap of {MAX_PICKS_CAP_DEFAULT}")
    except Exception as e:
        logger.warning(f"⚠️  Warning: Could not load K-AUTO map: {e}")
        logger.info(f"    Using default cap of {MAX_PICKS_CAP_DEFAULT} for all slots")
    
    logger.info("-" * 60)
    
    project_root = Path(__file__).resolve().parent
    processed = 0
    
    config_payload = {
        "mode": "future",
        "from": start_date.strftime("%Y-%m-%d"),
        "to": end_date.strftime("%Y-%m-%d"),
        "k_auto_map": k_auto_map,
        "scripts": SCRIPT_ORDER,
    }
    config_hash = _compute_config_hash(config_payload)
    run_metadata = {
        "mode": "future",
        "from": start_date.strftime("%Y-%m-%d"),
        "to": end_date.strftime("%Y-%m-%d"),
        "seed": None,
        "config_hash": config_hash,
    }

    # Process each day
    for day_num in range(total_days):
        prediction_date = start_date + dt.timedelta(days=day_num)
        
        # Skip month-end dates
        if _is_month_end(prediction_date):
            logger.info(f"SKIP: {prediction_date.strftime('%d-%m-%y')} is month-end")
            continue
        
        logger.info(f"\n📅 Day {day_num+1}/{total_days}: Generating predictions for {prediction_date.strftime('%d-%m-%y')}")
        logger.info("-" * 50)
        
        # Run all scripts
        cutoff_date = prediction_date - dt.timedelta(days=1)
        _run_scripts_quietly(
            project_root,
            cutoff_date=cutoff_date,
            scr_timeout=scr_timeout,
            scr_retries=scr_retries,
            scr2_date=prediction_date,
        )
        
        try:
            # Load and apply K-AUTO cap to shortlist
            shortlist = _load_shortlist()
            if not shortlist.empty:
                # Apply K-AUTO cap
                shortlist, trim_notes = _apply_max_cap(shortlist, k_auto_map)
                
                # Save the capped shortlist
                _save_shortlist_with_history(shortlist, prediction_date)
                date_folder = OUTPUT_DIR / prediction_date.strftime("%Y-%m-%d")
                metadata = {
                    **run_metadata,
                    "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                }
                _write_metadata(date_folder, metadata)
                
                # Print predictions
                candidates = _strongest_candidates(shortlist)
                logger.info("\n✅ STRONGEST ANDAR/BAHAR DIGITS")
                for slot_name, number in candidates:
                    tens = number // 10
                    ones = number % 10
                    logger.info(f"{slot_name} → {number:02d} (tens:{tens}, ones:{ones})")
                
                logger.info("\n📊 FINAL BET NUMBERS")
                bet_lines = _slot_bet_lines(shortlist, trim_notes)
                for line in bet_lines:
                    logger.info(line)
                
                # Show trimming info
                if trim_notes:
                    logger.info(f"\n🔧 Applied K-AUTO caps: {'; '.join(trim_notes)}")
                
                # Calculate expected stake
                total_picks = len(shortlist)
                expected_stake = total_picks * cfg.cost_per_unit
                logger.info(f"\n💰 Total Picks: {total_picks} | Expected Stake: ₹{expected_stake:.0f}")
                
                # Save daily report
                gate_cutoff = _resolve_ab_cutoff_date(prediction_date, ab_cutoff)
                gate_snapshot = compute_ab_gate_snapshot(gate_cutoff)
                _generate_daily_report(
                    prediction_date,
                    shortlist,
                    trim_notes,
                    k_auto_map,
                    gate_snapshot=gate_snapshot,
                    cutoff_date=gate_cutoff,
                )
                logger.info(f"✅ Predictions saved for {prediction_date.strftime('%d-%m-%y')}")
                processed += 1
            else:
                logger.warning(f"⚠️  No predictions generated for {prediction_date.strftime('%d-%m-%y')}")
                
        except FileNotFoundError as e:
            logger.info(f"❌ Error: {e}")
        except Exception as e:
            logger.info(f"❌ Unexpected error: {e}")
    
    logger.info(f"\n{'='*60}")
    logger.info("FUTURE PREDICTIONS GENERATION COMPLETE!")
    logger.info(f"Successfully generated predictions for {processed} days")
    logger.info(f"Check folder: {OUTPUT_DIR}")
    logger.info(f"{'='*60}")
    if not dry_run:
        _autopush_if_needed(
            f"Future predictions update {start_date.strftime('%Y%m%d')}-{end_date.strftime('%Y%m%d')}"
        )


def _safe_autopush():
    try:
        repo_root = os.path.dirname(os.path.abspath(__file__))
        candidate = os.path.join(repo_root, "tools", "git_autopush_only.cmd")
        if os.path.exists(candidate):
            logger.info("Running auto-push: tools\\git_autopush_only.cmd")
            subprocess.run([candidate], check=False, creationflags=0)
        else:
            logger.warning("auto-push skipped (tools\\git_autopush_only.cmd not found)")
    except Exception as e:
        logger.warning(f"auto-push failed: {e}")

if __name__ == "__main__":
    try:
        main()
    finally:
        _safe_autopush()
