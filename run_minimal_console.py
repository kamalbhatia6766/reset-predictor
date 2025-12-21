"""
Minimal console runner that executes SCR1–SCR9 quietly, then prints a concise
summary (shortlist path, strongest Andar/Bahar digits) plus a compact Bet PnL
snapshot with layer performance metrics.
"""
from __future__ import annotations

import calendar
import datetime as dt
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
import json

import pandas as pd

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


def _run_scripts_quietly(project_root: Path) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("w", encoding="utf-8") as log:
        for script in SCRIPT_ORDER:
            subprocess.run(
                [sys.executable, str(project_root / script)],
                stdout=log,
                stderr=log,
                check=True,
                cwd=project_root,
            )


def _load_shortlist() -> pd.DataFrame:
    xlsx_path = OUTPUT_DIR / "scr9_shortlist.xlsx"
    csv_path = OUTPUT_DIR / "scr9_shortlist.csv"

    if xlsx_path.exists():
        return pd.read_excel(xlsx_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)

    raise FileNotFoundError("SCR9 shortlist not found; expected scr9_shortlist.xlsx or scr9_shortlist.csv")


def _save_shortlist_with_history(shortlist: pd.DataFrame, prediction_date: dt.date) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Persist the latest files for compatibility
    shortlist.to_excel(OUTPUT_DIR / "scr9_shortlist.xlsx", index=False)

    # Date-stamped copy to keep per-day predictions intact
    dated_dir = OUTPUT_DIR / prediction_date.strftime("%Y-%m-%d")
    dated_dir.mkdir(parents=True, exist_ok=True)
    shortlist.to_excel(dated_dir / "scr9_shortlist.xlsx", index=False)


def _load_prediction_history() -> tuple[pd.DataFrame, bool]:
    if not HISTORY_PATH.exists():
        return pd.DataFrame(), True

    history = pd.read_csv(HISTORY_PATH)
    if "date" in history.columns:
        history["date"] = pd.to_datetime(history["date"], errors="coerce").dt.date
    return history, False


def _append_history(history: pd.DataFrame, new_rows: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([history, new_rows], ignore_index=True)
    if not combined.empty:
        combined["date"] = pd.to_datetime(combined["date"], errors="coerce").dt.date
        combined = combined.dropna(subset=["date", "slot", "number"])
        combined = combined.drop_duplicates(subset=["date", "slot", "number"], keep="last")
    combined.to_csv(HISTORY_PATH, index=False)
    return combined


def _is_month_end(value: dt.date) -> bool:
    return value.day == calendar.monthrange(value.year, value.month)[1]


def _next_non_month_end(value: dt.date) -> dt.date:
    next_day = value
    while _is_month_end(next_day):
        next_day += dt.timedelta(days=1)
    return next_day


def _latest_non_month_end_date(dates: List[dt.date]) -> dt.date:
    valid = [d for d in dates if not _is_month_end(d)]
    if not valid:
        raise ValueError("No non-month-end dates found in results")
    return max(valid)


def _strongest_candidates(df: pd.DataFrame) -> List[Tuple[str, int]]:
    strongest: List[Tuple[str, int]] = []
    slot_order = SLOT_NAME_TO_ID
    for slot_name, group in sorted(df.groupby("slot"), key=lambda item: slot_order.get(item[0], 99)):
        pick = group.sort_values("rank").iloc[0]
        strongest.append((slot_name, int(pick["number"])))
    return strongest


def _slot_bet_lines(df: pd.DataFrame, trimmed_notes: List[str]) -> List[str]:
    lines: List[str] = []
    for slot_name in SLOT_NAME_TO_ID.keys():
        group = df[df["slot"] == slot_name]
        if group.empty:
            continue
        numbers = group.sort_values("rank")["number"].astype(int).tolist()
        padded = " ".join(f"{n:02d}" for n in numbers)
        
        # Find trimming note for this slot
        trim_note = ""
        for note in trimmed_notes:
            if note.startswith(slot_name):
                # Extract original count from note like "FRBD: trimmed 35→20"
                if "→" in note:
                    original = note.split("→")[0].split()[-1]
                    # FIX C: Clean double parentheses
                    trim_note = f", trimmed {original}→{len(numbers)}"
                    break
        
        lines.append(f"{slot_name} ({len(numbers)}{trim_note}): {padded}")
    return lines


def _apply_max_cap(shortlist: pd.DataFrame, k_auto_map: Dict[int, int]) -> tuple[pd.DataFrame, List[str]]:
    if shortlist.empty:
        return shortlist, []

    trimmed = shortlist.copy()
    notes: List[str] = []

    for slot_name in SLOT_NAME_TO_ID.keys():
        group = shortlist[shortlist["slot"] == slot_name]
        if group.empty:
            continue
        slot_id = SLOT_NAME_TO_ID.get(slot_name, None)
        cap = k_auto_map.get(slot_id, MAX_PICKS_CAP_DEFAULT)
        if len(group) <= cap:
            continue

        def _strength(row: pd.Series) -> float:
            sources_raw = str(row.get("sources", ""))
            scripts = [s.strip() for s in sources_raw.replace(";", ",").split(",") if s.strip()]
            strength = len(scripts) if scripts else 1
            score = float(row.get("score", 0.0)) if pd.notna(row.get("score")) else 0.0
            votes = float(row.get("votes", 0.0)) if pd.notna(row.get("votes")) else 0.0
            rank_val = float(row.get("rank", 999)) if pd.notna(row.get("rank")) else 999.0
            return strength * 1000 + score + votes + max(0.0, 200.0 - rank_val)

        ranked = group.assign(strength=group.apply(_strength, axis=1)).sort_values(
            ["strength", "rank", "score"], ascending=[False, True, False]
        )
        keep = ranked.head(cap)
        trimmed = trimmed.drop(group.index.difference(keep.index))
        notes.append(f"{slot_name}: trimmed {len(group)}→{cap} (stronger = multi-script / higher score)")

    return trimmed, notes


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


def _format_digit_summary(candidates: List[Tuple[str, int]]) -> str:
    parts: List[str] = []
    for slot_name in SLOT_NAME_TO_ID.keys():
        match = next((num for name, num in candidates if name == slot_name), None)
        if match is None:
            continue
        tens = match // 10
        ones = match % 10
        parts.append(f"{slot_name} → {match:02d} (tens:{tens}, ones:{ones})")
    return "\n".join(parts)


def _prepare_predictions(df: pd.DataFrame, eval_date: dt.date) -> pd.DataFrame:
    if df.empty or "slot" not in df.columns:
        return pd.DataFrame(columns=["date", "slot", "number", "tier", "in_top", "rank", "votes", "score", "sources"])

    work = df.copy()
    work["date"] = eval_date
    work["slot"] = work["slot"].map(SLOT_NAME_TO_ID)
    keep = ["date", "slot", "number", "tier", "in_top", "rank", "votes", "score", "sources"]
    cols = [c for c in keep if c in work.columns]
    return work[cols]


def _parse_gate_map(ab_gate_lines: List[str]) -> Dict[str, bool]:
    """Parse gate table lines into a slot -> gate status map."""

    gate_map: Dict[str, bool] = {}
    for line in ab_gate_lines:
        parts = line.split()
        if len(parts) < 2:
            continue
        slot, status = parts[0], parts[1]
        if slot in SLOT_NAME_TO_ID:
            gate_map[slot] = status.strip().upper() == "[ON]"
    return gate_map


def _weight_sort_columns(group: pd.DataFrame) -> Tuple[List[str], List[bool]]:
    """Return (columns, ascending flags) expressing heaviness preference."""
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
        if "date" in history_df.columns:
            history_df["date"] = pd.to_datetime(history_df["date"], errors="coerce").dt.date
        prev_day_df = history_df[history_df["date"] == latest_date]

        if not prev_day_df.empty and "slot" in prev_day_df.columns:
            for slot_id in SLOT_NAME_MAP.keys():
                slot_group = prev_day_df[prev_day_df["slot"] == slot_id]
                if slot_group.empty:
                    continue
                sort_cols, ascending = _weight_sort_columns(slot_group)
                heaviest = slot_group.sort_values(by=sort_cols, ascending=ascending).iloc[0]
                prev_day_andars[slot_id] = heaviest["number"] // 10
                prev_day_bahars[slot_id] = heaviest["number"] % 10

    # FIX A & I: Add AB columns and fix totals
    lines = ["💰 YESTERDAY'S P&L ({})".format(latest_date.strftime("%d-%m-%y"))]
    lines.append("Slot    Result  Picks  Stake  Return   P&L     ROI      AB    AB P&L")
    lines.append("----    ------  -----  -----  ------   ----    ---      --    ------")
    
    total_picks = 0  # FIX A: Add total picks counter
    total_stake = 0
    total_return = 0
    total_pnl = 0
    total_ab_pnl = 0  # FIX I: Add AB P&L total

    for slot_id in slots:
        slot_df = latest_day_df[latest_day_df["slot"] == slot_id]
        slot_name = SLOT_NAME_MAP.get(slot_id, str(slot_id))
        actual_val = actual_map.get(slot_id)
        actual_display = "XX" if actual_val is None else f"{int(actual_val):02d}"

        picks = len(slot_df)
        total_picks += picks  # FIX A: Track total picks
        stake = slot_df["cost"].sum() if not slot_df.empty else 0.0
        returns = slot_df["payout"].sum() if not slot_df.empty else 0.0
        pnl = slot_df["pnl"].sum() if not slot_df.empty else 0.0
        roi = (pnl / stake * 100) if stake else 0.0

        total_stake += stake
        total_return += returns
        total_pnl += pnl

        hit_rows = slot_df[slot_df["hit"]]
        # FIX B: Replace emojis with ASCII
        result_symbol = "[HIT]" if not hit_rows.empty else "[MISS]"
        
        # FIX I: Add AB columns (previous-day picks)
        ab_status = "-"
        ab_pnl_val = 0

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
            f"{slot_name:<6}  {actual_display:<6}  {picks:>5}  {stake:>5.0f}  {returns:>6.0f}  {pnl:>+5.0f}  {roi:>+5.0f}%  {ab_status:>4}  {ab_pnl_display}  {result_symbol}"
        )

    # Add totals - FIX A: Show total picks properly
    lines.append("-" * 70)
    total_roi = (total_pnl / total_stake * 100) if total_stake else 0.0
    lines.append(
        f"Total:           {total_picks:>5}  {total_stake:>5.0f}  {total_return:>6.0f}  "
        f"{total_pnl:>+5.0f}  {total_roi:>+5.0f}%         {total_ab_pnl:>+4.0f}"
    )

    return "\n".join(lines)


def _format_rollups(combined_totals: pd.DataFrame) -> List[str]:
    lines: List[str] = []
    
    # FIX H: Replace progress bar with signal
    def _signal(roi: float) -> str:
        if roi > 0.15:
            return "[GOOD]"
        elif roi > 0:
            return "[OK]"
        else:
            return "[BAD]"

    for name in ("day", "7d", "month", "cumulative"):
        row = combined_totals[combined_totals["window"] == name]
        if row.empty:
            continue
        
        r = row.iloc[0]
        roi_display = f"{r['roi']:+.1%}"
        
        # Format stake in thousands if large
        stake_display = f"{r['stake']/1000:.1f}K" if r['stake'] >= 1000 else f"{r['stake']:.0f}"
        
        signal = _signal(r['roi'])
        lines.append(f"{name.title():<11} {signal:<7} Stake: {stake_display:<6} P&L: {r['pnl']:>+7.0f} ROI: {roi_display}")

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
    
    # Add skip lines if any
    if skip_lines:
        parts.extend(skip_lines)
        parts.append("")
    
    # Add P&L table
    parts.append(pnl_table)
    parts.append("")
    
    # Add day top-k strategy
    if day_topk_lines:
        parts.append(day_topk_lines[0])
        parts.append("")
    
    # Add Andar/Bahar gate
    if ab_gate_lines:
        parts.extend(ab_gate_lines)
        parts.append("")
    
    # Add Rank buckets
    if rank_lines:
        parts.extend(rank_lines)
        parts.append("")
    
    # Add Tag ROI
    if tag_lines:
        parts.extend(tag_lines)
        parts.append("")
    
    # Add Cross-slot hits
    if cross_slot_lines:
        parts.extend(cross_slot_lines)
        parts.append("")
    
    # Add Hero/Avoid
    if hero_lines:
        parts.extend(hero_lines)
        parts.append("")
    
    # Add Rollups
    if rollups:
        parts.append("📈 PERFORMANCE ROLLUPS")
        parts.append("Period      Status  Stake     P&L        ROI")
        parts.append("-------     ------  -----     ----       ---")
        parts.extend(rollups)
        parts.append("")
    
    # Add any remaining notes
    if notes:
        parts.append("📝 NOTES")
        for note in notes:
            parts.append(f"• {note}")
    
    return "\n".join(parts)


def _write_text(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def _prompt_date(raw: str, fallback: dt.date) -> dt.date | None:
    try:
        return dt.datetime.strptime(raw or fallback.strftime("%d-%m-%y"), "%d-%m-%y").date()
    except ValueError:
        print("SKIP: invalid date format; expected DD-mm-yy")
        return None


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
    
    # FIX B: Replace emojis with ASCII
    freshness = "[FRESH]" if not needs_rebuild else "[STALE]"
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
            # FIX B: Replace emojis with ASCII
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


def main() -> None:
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
        print("-" * 50)

        _run_scripts_quietly(project_root)

        effective_dates = build_effective_dates(window_start, window_end, available_dates=result_dates)

        bet_rows = load_clean_bet_rows(window_start, window_end, cfg)
        rank_lines, k_auto_map = format_rank_bucket_windows(bet_rows, effective_dates)
        topk_lines, day_topk_lines = format_topk_profit(
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
        
        # Print predictions section
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

        # Evaluate only for dates where actual results exist
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
        
        # Get hero lines
        hero_lines = []
        if prebuilt_metrics:
            hero_lines = format_hero_weakest(prebuilt_metrics, min_bets=20)
        
        stake_line = _stake_summary(cfg)

        present_dates = history_for_results["date"].dropna().tolist()
        skip_reason = "missing prediction files" if prediction_files_missing else "no predictions logged"
        skip_range_lines = _collapse_date_gaps(effective_dates, present_dates, skip_reason)

        # Filter empty lines
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

    print("\n" + "="*50)
    print("DAILY REPORT")
    print("="*50)
    print(daily_report_body)
    print("\n✅ Daily report saved to: reports/daily_report_auto.txt")


if __name__ == "__main__":
    main()
