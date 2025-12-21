"""
Compact profit & loss tracker with automatic andar/bahar picks.

Key behaviors
- Ingest a prediction set (date, slot, number, optional stake and weights).
- Auto-derive andar/bahar digits per slot from the heaviest-ranked number when
  explicit digits are not supplied. This matches the request that "jo number
  sabse bhari ho wo andar/bahar ke liye maana jaye" when SCR9 emits many bets.
- Join with real results from ``number prediction learn.xlsx`` and compute
  hits, payouts, and P&L windows (day/7d/month/cumulative).
- Track profit-adjacent signals (near-miss ±1, mirror, cross-slot/day) plus
  pack/family tags (S40, 164950) and add hit rank/top-N notes for reporting.
- Keep the rendered output short so it can be shared in chat without clutter.

Usage (example):
    from bet_pnl_tracker import compute_pnl_report, render_compact_report
    report = compute_pnl_report(predictions_df)
    print(render_compact_report(report))

Prediction schema expectations:
- Required: ``date``, ``slot`` (1-4), ``number`` (0-99)
- Optional: ``stake`` (default 1.0), ``score``, ``votes``, ``rank``,
  ``in_top`` (used to choose the "heaviest" number per slot), and optional
  ``andar``/``bahar`` digits (0-9). Extra columns are preserved.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import json
import sys
import argparse

import pandas as pd

from quant_data_core import load_results_dataframe

SLOT_NAME_MAP = {1: "FRBD", 2: "GZBD", 3: "GALI", 4: "DSWR"}
ANDAR_ALIASES = ("andar", "andar_digit")
BAHAR_ALIASES = ("bahar", "bahar_digit")
S40_PACK = {
    0,
    6,
    7,
    9,
    15,
    16,
    18,
    19,
    24,
    25,
    27,
    28,
    33,
    34,
    36,
    37,
    42,
    43,
    45,
    46,
    51,
    52,
    54,
    55,
    60,
    61,
    63,
    64,
    70,
    72,
    73,
    79,
    81,
    82,
    88,
    89,
    90,
    91,
    97,
    98,
}
FAMILY_164950_DIGITS = {0, 1, 4, 5, 6, 9}
PREBUILT_DIR = "reports/prebuilt_metrics"
MAX_PICKS_CAP_DEFAULT = 35


def _check_window_sufficiency(effective_dates: List[date], window_days: Optional[int]) -> tuple[bool, str]:
    """Check if we have enough data for window"""
    if not effective_dates:
        return False, "NO DATA"
    
    if window_days is None:  # Full window always considered sufficient
        return True, ""
    
    date_count = len(effective_dates)
    if date_count < window_days:
        return False, f"INSUFFICIENT (<{window_days} days; have {date_count})"
    return True, ""


def _log_skip(reason: str) -> str:
    message = f"SKIP: {reason}"
    print(message)
    return message


def _log_collapsed(reason_prefix: str, dates: List[date]) -> None:
    if not dates:
        return
    dates = sorted(set(dates))
    start, end = dates[0], dates[-1]
    if start == end:
        _log_skip(f"{reason_prefix} {start}")
    else:
        _log_skip(f"{reason_prefix} {start}→{end}")


def _available_result_dates() -> List[date]:
    results_df = load_results_dataframe()
    if results_df.empty:
        return []
    results_df["DATE"] = pd.to_datetime(results_df["DATE"], errors="coerce").dt.date
    return [d for d in results_df["DATE"].dropna().tolist()]


def build_effective_dates(start_date: date, end_date: date, available_dates: Optional[Iterable[date]] = None) -> List[date]:
    """Single source of truth for aligned dates with month-end skip + availability."""

    if start_date > end_date:
        start_date, end_date = end_date, start_date

    available_set = set(available_dates) if available_dates is not None else set(_available_result_dates())

    all_dates: List[date] = []
    current = start_date
    while current <= end_date:
        ts = pd.Timestamp(current)
        if not ts.is_month_end:
            if not available_set or current in available_set:
                all_dates.append(current)
        current += timedelta(days=1)

    return sorted(all_dates)


def _read_prediction_history() -> pd.DataFrame:
    path = Path("predictions/deepseek_scr9/scr9_shortlist_history.csv")
    if not path.exists():
        _log_skip("scr9_shortlist_history.csv missing")
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    if "slot" in df.columns:
        df["slot"] = pd.to_numeric(df["slot"], errors="coerce").astype("Int64")
    return df


def _pick_script_label(row: pd.Series) -> str:
    if "script" in row and pd.notna(row["script"]):
        return str(row["script"])
    sources = row.get("sources")
    if isinstance(sources, str) and sources.strip():
        parts = [p.strip() for p in sources.split(",") if p.strip()]
        if len(parts) == 1 and parts[0]:
            return parts[0]
    return "scr9_merged"


def _collect_tags(row: pd.Series) -> str:
    tags: List[str] = []
    for col in ("tags", "sources"):
        value = row.get(col)
        if isinstance(value, str) and value.strip():
            for chunk in value.replace(";", ",").split(","):
                if chunk.strip():
                    tags.append(chunk.strip())
    return ";".join(sorted(set(tags))) if tags else ""


def load_clean_bet_rows(start_date: date, end_date: date, cfg: "PnLConfig") -> pd.DataFrame:
    """Reusable raw rows builder with skip-guarded IO."""

    result_dates = _available_result_dates()
    if not result_dates:
        _log_skip("results file empty or unreadable")
        return pd.DataFrame()

    history = _read_prediction_history()
    if history.empty:
        _log_skip("no prediction history available")
        return pd.DataFrame()

    effective_dates = build_effective_dates(start_date, end_date, available_dates=result_dates)
    if not effective_dates:
        _log_skip("no effective dates in requested range")
        return pd.DataFrame()

    results_df = load_results_dataframe()
    results_df["DATE"] = pd.to_datetime(results_df["DATE"], errors="coerce").dt.date

    rows: List[Dict[str, object]] = []
    missing_preds: List[date] = []
    missing_results: List[date] = []
    missing_slot: Dict[str, List[date]] = {}

    for day in effective_dates:
        day_preds = history[history["date"] == day]
        if day_preds.empty:
            missing_preds.append(day)
            continue

        actual_row = results_df[results_df["DATE"] == day]
        if actual_row.empty:
            missing_results.append(day)
            continue

        actual_row = actual_row.iloc[0]

        for slot_id, slot_name in SLOT_NAME_MAP.items():
            actual_val = actual_row.get(slot_name)
            slot_preds = day_preds[day_preds["slot"] == slot_id]
            if slot_preds.empty:
                missing_slot.setdefault(slot_name, []).append(day)
                continue

            for _, row in slot_preds.iterrows():
                number = row.get("number")
                if pd.isna(number):
                    continue
                hit = False
                if pd.notna(actual_val):
                    hit = int(number) == int(actual_val)

                payout = cfg.cost_per_unit * cfg.payout_per_unit if hit else 0.0
                top_rank = row.get("rank") if hit else None

                rows.append(
                    {
                        "date": day,
                        "slot": int(slot_id),
                        "script": _pick_script_label(row),
                        "top_n": top_rank if pd.notna(top_rank) else None,
                        "hit": bool(hit),
                        "cost": cfg.cost_per_unit,
                        "payout": payout,
                        "tags": _collect_tags(row),
                        "actual": int(actual_val) if pd.notna(actual_val) else None,
                        "number": int(number),
                        "rank": int(row["rank"]) if "rank" in row and pd.notna(row["rank"]) else None,
                    }
                )

    _log_collapsed("no predictions for", missing_preds)
    _log_collapsed("no results for", missing_results)
    for slot_name, days in missing_slot.items():
        _log_collapsed(f"no predictions for {slot_name} on", days)

    return pd.DataFrame(rows)


def _select_window_dates(rows: pd.DataFrame, window_days: Optional[int], effective_dates: List[date]) -> List[date]:
    if not effective_dates:
        unique_dates = sorted({d for d in pd.to_datetime(rows["date"], errors="coerce").dt.date.dropna().tolist()})
        effective_dates = unique_dates
    if window_days is None:
        return effective_dates
    return effective_dates[-window_days:]


def aggregate_metrics(
    rows: pd.DataFrame, window_days: Optional[int] = None, effective_dates: Optional[List[date]] = None
) -> pd.DataFrame:
    columns = ["slot", "script", "bets", "hits", "hit_rate", "stake", "return", "pnl", "roi", "avg_hit_rank"]
    if rows.empty:
        return pd.DataFrame(columns=columns)

    work = rows.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce").dt.date
    dates = _select_window_dates(work, window_days, effective_dates or [])
    if dates:
        work = work[work["date"].isin(set(dates))]

    if work.empty:
        return pd.DataFrame(columns=columns)

    group = work.groupby(["slot", "script"], dropna=False)
    agg = group.agg(
        bets=("hit", "size"),
        hits=("hit", "sum"),
        hit_rate=("hit", "mean"),
        stake=("cost", "sum"),
        return_=("payout", "sum"),
    )

    agg.rename(columns={"return_": "return"}, inplace=True)
    agg["hits"] = agg["hits"].astype(int)
    agg["hit_rate"] = agg["hit_rate"].fillna(0.0)
    agg["pnl"] = agg["return"] - agg["stake"]
    agg["roi"] = agg["pnl"] / agg["stake"].replace(0, pd.NA)
    agg["roi"] = agg["roi"].where(agg["stake"] != 0, pd.NA)

    agg["avg_hit_rank"] = group.apply(
        lambda g: g.loc[g["hit"], "top_n"].dropna().mean() if g["hit"].any() else None,
        include_groups=False,
    )

    agg = agg.reset_index()

    return agg[columns]


def _window_frames_equal(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    if left.empty and right.empty:
        return True
    if left.empty or right.empty:
        return False
    sort_cols = ["slot", "script", "bets", "hits", "roi", "pnl"]
    return (
        left.sort_values(sort_cols).reset_index(drop=True)
        .fillna("__NA__")
        .equals(right.sort_values(sort_cols).reset_index(drop=True).fillna("__NA__"))
    )


def _roi_label(metrics_by_window: Dict[str, pd.DataFrame], slot_id: int, script: str) -> str:
    parts: List[str] = []
    for key in ("7d", "30d", "full"):
        df = metrics_by_window.get(key)
        if df is None or df.empty:
            continue
        row = df[(df["slot"] == slot_id) & (df["script"] == script)]
        if row.empty:
            continue
        bets = int(row.iloc[0]["bets"])
        roi_val = row.iloc[0]["roi"]
        roi_display = "SKIP" if bets == 0 else ("n/a" if pd.isna(roi_val) else f"{roi_val:+.0%}")
        parts.append(f"{key}:{roi_display} ({bets} bets)")
    return " / ".join(parts) if parts else "n/a"


def format_hero_weakest(metrics_by_window: Dict[str, pd.DataFrame], min_bets: int = 20) -> List[str]:
    """Return compact hero/weakest script table with max 2 HERO + 1 AVOID per slot."""
    lines: List[str] = []
    available = {k: v for k, v in metrics_by_window.items() if v is not None and not v.empty}
    if not available:
        return lines

    # Use 30D metrics for decisions
    df_30d = available.get("30d")
    if df_30d is None or df_30d.empty:
        # Fallback to full if 30D not available
        df_30d = available.get("full")
        if df_30d is None or df_30d.empty:
            return lines

    lines.append("HERO/AVOID SCRIPTS (30D)")
    lines.append("Slot   Script      Bets  ROI(30D)  Status      Note")
    lines.append("----   ------      ----  --------  ------      ----")

    for slot_id, slot_name in SLOT_NAME_MAP.items():
        slot_metrics = df_30d[df_30d["slot"] == slot_id]
        if slot_metrics.empty:
            continue

        # Sort by ROI (descending) and bets (descending)
        slot_metrics = slot_metrics.sort_values(["roi", "bets"], ascending=[False, False])

        # Get top HERO scripts (ROI >= 0, bets >= min_bets)
        hero_candidates = slot_metrics[
            (slot_metrics["roi"] >= 0) & (slot_metrics["bets"] >= min_bets)
        ].head(2)
        
        # Get top AVOID script (ROI < 0, bets >= min_bets)
        avoid_candidates = slot_metrics[
            (slot_metrics["roi"] < 0) & (slot_metrics["bets"] >= min_bets)
        ].head(1)

        # Process HERO scripts
        hero_count = 0
        for _, row in hero_candidates.iterrows():
            script_name = str(row["script"])
            bets = int(row["bets"])
            roi_val = row["roi"]
            roi_display = f"{roi_val:+.0%}" if not pd.isna(roi_val) else "n/a"
            
            if bets < min_bets:
                status = "WATCH"
                note = f"<{min_bets} bets"
            else:
                status = "HERO"
                note = "Strong"
            
            lines.append(f"{slot_name:<6} {script_name:<11} {bets:>4}  {roi_display:>8}  {status:<11} {note}")
            hero_count += 1

        # Process AVOID script
        for _, row in avoid_candidates.iterrows():
            script_name = str(row["script"])
            bets = int(row["bets"])
            roi_val = row["roi"]
            roi_display = f"{roi_val:+.0%}" if not pd.isna(roi_val) else "n/a"
            
            if bets < min_bets:
                status = "WATCH"
                note = f"<{min_bets} bets"
            else:
                status = "AVOID"
                note = "Loss-making"
            
            lines.append(f"{slot_name:<6} {script_name:<11} {bets:>4}  {roi_display:>8}  {status:<11} {note}")

        # If no scripts met criteria, add a placeholder
        if hero_candidates.empty and avoid_candidates.empty:
            lines.append(f"{slot_name:<6} {'--':<11} {'--':>4}  {'--':>8}  {'NO DATA':<11} {'--':<10}")

    return lines


def _rank_buckets(
    rows: pd.DataFrame, window_days: int | None = None, effective_dates: Optional[List[date]] = None
) -> Dict[int, Counter]:
    if rows.empty:
        return {}

    work = rows.copy()
    window_dates = _select_window_dates(work, window_days, effective_dates or [])
    if window_dates:
        work = work[work["date"].isin(set(window_dates))]

    work = work[work["hit"] & work["top_n"].notna()]
    buckets: Dict[int, Counter] = defaultdict(Counter)
    for _, row in work.iterrows():
        slot = int(row["slot"])
        rank = int(row["top_n"])
        if rank <= 10:
            key = "1-10"
        elif rank <= 15:
            key = "11-15"
        elif rank <= 20:
            key = "16-20"
        elif rank <= 25:
            key = "21-25"
        elif rank <= 30:
            key = "26-30"
        elif rank <= 40:
            key = "31-40"
        else:
            key = "41+"
        buckets[slot][key] += 1
    return buckets


def _append_tag_flags(rows: pd.DataFrame) -> pd.DataFrame:
    work = rows.copy()
    work["tags"] = work.get("tags", "").fillna("").astype(str)
    work["_tags_lower"] = work["tags"].str.lower()

    def _has_keyword(series: pd.Series, keywords: Tuple[str, ...]) -> pd.Series:
        return series.apply(lambda s: any(k in s for k in keywords))

    work["tag_s40"] = work["number"].apply(lambda n: int(n) in S40_PACK)
    work["tag_164950"] = work["number"].apply(_is_family_164950_member)
    work["tag_both"] = work["tag_s40"] & work["tag_164950"]
    work["tag_mirror"] = _has_keyword(work["_tags_lower"], ("mirror",))
    work["tag_neighbour"] = _has_keyword(work["_tags_lower"], ("neighbour", "neighbor"))

    return work.drop(columns=["_tags_lower"])


def _auto_k_from_buckets(bucket_counter: Counter) -> int:
    total_hits = sum(bucket_counter.values())
    if total_hits == 0:
        return 20
    le20 = sum(bucket_counter[k] for k in ("1-10", "11-15", "16-20"))
    mid = sum(bucket_counter[k] for k in ("21-25", "26-30", "31-40"))
    over35 = bucket_counter["41+"]
    if le20 / total_hits >= 0.6:
        return 20
    if mid / total_hits >= 0.6:
        return 30 if bucket_counter["21-25"] + bucket_counter["26-30"] >= bucket_counter["31-40"] else 35
    if over35 / total_hits >= 0.4:
        return 40
    return 35


def _format_rank_buckets(
    rows: pd.DataFrame, window_days: int | None = None, effective_dates: Optional[List[date]] = None
) -> List[str]:
    buckets = _rank_buckets(rows, window_days, effective_dates)
    lines: List[str] = []
    for slot_id, slot_name in SLOT_NAME_MAP.items():
        counter = buckets.get(slot_id, Counter())
        if not counter:
            lines.append(f"{slot_name} RankBuckets: SKIP no hits")
            continue
        parts = [f"{k}:{counter.get(k,0)}" for k in ("1-10", "11-15", "16-20", "21-25", "26-30", "31-40", "41+")]
        k_auto = _auto_k_from_buckets(counter)
        lines.append(f"{slot_name} RankBuckets: {' '.join(parts)} => K-AUTO={k_auto}")
    return lines


def format_rank_bucket_windows(rows: pd.DataFrame, effective_dates: Optional[List[date]] = None) -> Tuple[List[str], Dict[int, int]]:
    """Return rank bucket summary tables for 7D/30D/full windows."""

    # FIX J: Check window sufficiency
    sufficient_7d, reason_7d = _check_window_sufficiency(effective_dates, 7)
    sufficient_30d, reason_30d = _check_window_sufficiency(effective_dates, 30)

    lines: List[str] = []
    k_auto_map: Dict[int, int] = {}

    window_defs = [(7, "7D"), (30, "30D"), (None, "Full")]
    bucket_by_window: Dict[str, Dict[int, Counter]] = {}

    for window_days, label in window_defs:
        bucket_by_window[label] = _rank_buckets(rows, window_days=window_days, effective_dates=effective_dates)

    # FIX E: Prefer 30D first, then 7D, then Full
    display_label = None
    for preferred_label in ("30D", "7D", "Full"):
        if any(bucket_by_window.get(preferred_label, {}).values()):
            display_label = preferred_label
            break

    # FIX J: Show insufficient message if 30D selected but insufficient
    if display_label == "30D" and not sufficient_30d:
        return [f"RankBuckets: {reason_30d}"], {}

    # Preserve legacy preference for deriving K-AUTO caps: 30D -> 7D -> Full.
    for label in ("30D", "7D", "Full"):
        for slot_id, counter in bucket_by_window.get(label, {}).items():
            if slot_id not in k_auto_map and counter:
                k_auto_map[slot_id] = _auto_k_from_buckets(counter)

    if not display_label:
        return ["RankBuckets: SKIP (no hits across windows)"], k_auto_map

    column_headers = ["Top10", "11-15", "16-20", "21-25", "26-30", "31-40", "K-AUTO"]
    header_line = f"{'Slot':<6}" + "".join(f"{h:>7}" for h in column_headers)

    lines.append(f"🔢 RANK BUCKETS & K-AUTO ({display_label})")
    lines.append(header_line)

    window_counters = bucket_by_window.get(display_label, {})
    for slot_id, slot_name in SLOT_NAME_MAP.items():
        counter = window_counters.get(slot_id, Counter())
        if counter:
            values = [counter.get(bucket, 0) for bucket in ("1-10", "11-15", "16-20", "21-25", "26-30", "31-40")]
            k_auto = k_auto_map.get(slot_id, _auto_k_from_buckets(counter))
            row = f"{slot_name:<6}" + "".join(f"{v:>7}" for v in values) + f"{k_auto:>8}"
        else:
            row = f"{slot_name:<6}{'SKIP':>7}"
        lines.append(row)

    return lines, k_auto_map


def _tag_roi(
    rows: pd.DataFrame, window_days: int | None = None, effective_dates: Optional[List[date]] = None
) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame(columns=["slot", "tag", "bets", "hits", "stake", "payout", "pnl", "roi"])

    work = _append_tag_flags(rows)
    work["date"] = pd.to_datetime(work["date"], errors="coerce").dt.date
    window_dates = _select_window_dates(work, window_days, effective_dates or [])
    if window_dates:
        work = work[work["date"].isin(set(window_dates))]

    if work.empty:
        return pd.DataFrame(columns=["slot", "tag", "bets", "hits", "stake", "payout", "pnl", "roi"])

    tag_columns = [
        ("S40", "tag_s40"),
        ("164950", "tag_164950"),
        ("both", "tag_both"),
        ("mirror", "tag_mirror"),
        ("neighbour", "tag_neighbour"),
    ]

    rows_out: List[Dict[str, object]] = []
    for slot_id, slot_group in work.groupby("slot"):
        for tag_name, col in tag_columns:
            tagged = slot_group[slot_group[col]]
            bets = len(tagged)
            stake = tagged["cost"].sum()
            payout = tagged["payout"].sum()
            pnl = payout - stake
            roi = pnl / stake if stake else pd.NA
            rows_out.append(
                {
                    "slot": int(slot_id),
                    "tag": tag_name,
                    "bets": bets,
                    "hits": int(tagged["hit"].sum()) if bets else 0,
                    "stake": stake,
                    "payout": payout,
                    "pnl": pnl,
                    "roi": roi,
                }
            )

    return pd.DataFrame(rows_out)


def _tag_booster_status(
    rows: pd.DataFrame, effective_dates: Optional[List[date]] = None
) -> Dict[int, Dict[str, Tuple[str, int]]]:
    """Decide booster ON/OFF per slot/tag using ROI and latest-day pick pressure."""

    work = _append_tag_flags(rows)
    work["date"] = pd.to_datetime(work["date"], errors="coerce").dt.date
    window_dates = _select_window_dates(work, None, effective_dates or [])
    latest_date = window_dates[-1] if window_dates else (pd.to_datetime(work["date"]).dt.date.max() if not work.empty else None)

    metrics = {
        "30d": _tag_roi(rows, window_days=30, effective_dates=effective_dates),
        "7d": _tag_roi(rows, window_days=7, effective_dates=effective_dates),
        "full": _tag_roi(rows, window_days=None, effective_dates=effective_dates),
    }

    decisions: Dict[int, Dict[str, Tuple[str, int]]] = defaultdict(dict)
    tags = ["S40", "164950", "both", "mirror", "neighbour"]

    for slot_id in SLOT_NAME_MAP.keys():
        latest_extra = {t: 0 for t in tags}
        if latest_date:
            slot_latest = work[(work["slot"] == slot_id) & (work["date"] == latest_date)]
            for t, col in (
                ("S40", "tag_s40"),
                ("164950", "tag_164950"),
                ("both", "tag_both"),
                ("mirror", "tag_mirror"),
                ("neighbour", "tag_neighbour"),
            ):
                latest_extra[t] = int(slot_latest[col].sum()) if not slot_latest.empty else 0

        for tag in tags:
            metric_row = None
            for key in ("30d", "7d", "full"):
                frame = metrics.get(key)
                if frame is not None and not frame.empty:
                    candidate = frame[(frame["slot"] == slot_id) & (frame["tag"] == tag)]
                    if not candidate.empty:
                        metric_row = candidate.iloc[0]
                        break

            decision = "[OFF]"
            if metric_row is not None:
                roi = metric_row.get("roi")
                stake = metric_row.get("stake", 0)
                if stake and pd.notna(roi) and roi > 0.10 and latest_extra[tag] <= 4:
                    decision = "[ON]"

            decisions[slot_id][tag] = (decision, latest_extra[tag])

    return decisions


def format_tag_roi(
    rows: pd.DataFrame, effective_dates: Optional[List[date]] = None, unit_stake: float = 1.0
) -> List[str]:
    """Compact tag ROI table focused on core packs and booster state."""

    # FIX J: Check window sufficiency
    sufficient, reason = _check_window_sufficiency(effective_dates, 30)
    if not sufficient:
        return [f"Tag ROI: {reason}"]

    if rows.empty:
        return ["Tag ROI: SKIP (no bet rows)"]

    windows = [(30, "30D"), (7, "7D"), (None, "Full")]
    metrics_by_window = {label: _tag_roi(rows, window_days=window_days, effective_dates=effective_dates) for window_days, label in windows}

    target_label = None
    target_metrics = None
    for _, label in windows:
        frame = metrics_by_window.get(label)
        if frame is not None and not frame.empty:
            target_label = label
            target_metrics = frame
            break

    if target_metrics is None:
        return ["Tag ROI: SKIP (no tag data)"]

    boosters = _tag_booster_status(rows, effective_dates)

    def _roi_display(slot_id: int, tag: str) -> str:
        row = target_metrics[(target_metrics["slot"] == slot_id) & (target_metrics["tag"] == tag)]
        if row.empty:
            return "SKIP"
        r = row.iloc[0]
        stake = r["stake"]
        roi_val = r["roi"]
        if not stake:
            return "SKIP"
        if pd.isna(roi_val):
            return "n/a"
        return f"{roi_val:+.0%}"

    # FIX B: Remove emojis
    header = f"TAG ROI ({target_label}) & BOOSTER"
    column_titles = ["Slot", "S40 ROI", "164950 ROI", "BOTH ROI", "Booster"]
    lines = [header, " | ".join(column_titles), "-" * 50]

    for slot_id, slot_name in SLOT_NAME_MAP.items():
        s40_roi = _roi_display(slot_id, "S40")
        family_roi = _roi_display(slot_id, "164950")
        both_roi = _roi_display(slot_id, "both")

        boost_state = boosters.get(slot_id, {})
        boost_on = any(boost_state.get(tag, ("[OFF]", 0))[0] == "[ON]" for tag in ("S40", "164950", "both"))
        boost_label = "[ON]" if boost_on else "[OFF]"
        
        # FIX F: Add booster reason
        if boost_label == "[OFF]":
            boost_label = f"[OFF] (rule: ROI>+10% AND picks≤4)"

        lines.append(f"{slot_name} | {s40_roi:>7} | {family_roi:>11} | {both_roi:>9} | {boost_label}")

    return lines


def _cross_slot_counts(rows: pd.DataFrame, effective_dates: Optional[List[date]] = None, lookback_days: int = 60) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame(columns=["pred_slot", "hit_slot", "hits"])

    work = rows.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce").dt.date
    window_dates = _select_window_dates(work, lookback_days, effective_dates or [])
    if window_dates:
        work = work[work["date"].isin(set(window_dates))]

    if work.empty:
        return pd.DataFrame(columns=["pred_slot", "hit_slot", "hits"])

    results_df = load_results_dataframe()
    if results_df.empty:
        _log_skip("results file missing or empty for cross-slot tracking")
        return pd.DataFrame(columns=["pred_slot", "hit_slot", "hits"])

    results_df["DATE"] = pd.to_datetime(results_df["DATE"], errors="coerce").dt.date
    result_dates = results_df["DATE"].dropna().tolist()
    if window_dates:
        allowed_dates = set(window_dates)
    else:
        if not result_dates:
            return pd.DataFrame(columns=["pred_slot", "hit_slot", "hits"])
        allowed_dates = set(build_effective_dates(min(result_dates), max(result_dates), available_dates=result_dates))

    results_df = results_df[results_df["DATE"].isin(allowed_dates)]
    if results_df.empty:
        return pd.DataFrame(columns=["pred_slot", "hit_slot", "hits"])

    per_date_actuals: Dict[date, Dict[int, int]] = {}
    for _, row in results_df.iterrows():
        actuals: Dict[int, int] = {}
        for slot_id, slot_name in SLOT_NAME_MAP.items():
            val = row.get(slot_name)
            if pd.notna(val):
                try:
                    actuals[int(slot_id)] = int(val)
                except (TypeError, ValueError):
                    continue
        per_date_actuals[row["DATE"]] = actuals

    counter: Counter[Tuple[int, int]] = Counter()
    for _, row in work.iterrows():
        dt_val = row.get("date")
        if pd.isna(dt_val):
            continue
        actuals = per_date_actuals.get(dt_val)
        if not actuals:
            continue
        try:
            predicted_number = int(row["number"])
            pred_slot = int(row["slot"])
        except (TypeError, ValueError):
            continue

        for actual_slot, actual_number in actuals.items():
            if predicted_number == actual_number:
                counter[(pred_slot, actual_slot)] += 1

    rows_out = [
        {"pred_slot": p, "hit_slot": h, "hits": count}
        for (p, h), count in counter.items()
        if count > 0
    ]
    return pd.DataFrame(rows_out)


def format_cross_slot_hits(rows: pd.DataFrame, effective_dates: Optional[List[date]] = None) -> List[str]:
    lines: List[str] = []
    counts = _cross_slot_counts(rows, effective_dates=effective_dates, lookback_days=60)
    if counts.empty:
        return ["CROSS-SLOT: No data (60D)"]

    lines.append("CROSS-SLOT HITS (60D)")
    for slot_id, slot_name in SLOT_NAME_MAP.items():
        slot_rows = counts[counts["pred_slot"] == slot_id]
        if slot_rows.empty:
            lines.append(f"{slot_name}: No cross-hits")
            continue

        top_hits = slot_rows.sort_values("hits", ascending=False).head(2)
        parts = [
            f"{SLOT_NAME_MAP.get(int(r['hit_slot']), r['hit_slot'])} ({int(r['hits'])})"
            for _, r in top_hits.iterrows()
        ]
        lines.append(f"{slot_name} → " + ", ".join(parts))

    return lines


def _topk_profit(row_slice: pd.DataFrame, actual: int, unit_stake: float) -> Dict[int, Dict[str, float]]:
    ranks = row_slice.sort_values("rank").reset_index(drop=True)
    result: Dict[int, Dict[str, float]] = {}
    for k in (10, 15, 20, 25, 30, 35, 40):
        picks = min(k, len(ranks))
        stake = picks * unit_stake
        if picks == 0:
            result[k] = {"stake": 0.0, "return": 0.0, "pnl": 0.0, "roi": 0.0, "hit": False, "rank": None}
            continue
        top_numbers = ranks.head(picks)
        hit_row = top_numbers[top_numbers["number"] == actual]
        if not hit_row.empty:
            ret = unit_stake * 90.0
            pnl = ret - stake
            roi = pnl / stake if stake else 0.0
            result[k] = {
                "stake": stake,
                "return": ret,
                "pnl": pnl,
                "roi": roi,
                "hit": True,
                "rank": int(hit_row.iloc[0]["rank"]) if "rank" in hit_row.columns else None,
            }
        else:
            pnl = -stake
            roi = pnl / stake if stake else 0.0
            result[k] = {"stake": stake, "return": 0.0, "pnl": pnl, "roi": roi, "hit": False, "rank": None}
    return result


def _format_topk(
    rows: pd.DataFrame, effective_dates: Optional[List[date]] = None, unit_stake: float = 1.0
) -> Tuple[List[str], List[str]]:
    if rows.empty:
        return ["TopK: SKIP (no shortlist data)"], []

    rows = rows.dropna(subset=["actual"]).copy()
    if rows.empty:
        return ["TopK: SKIP (no actual results)"], []

    window_dates = _select_window_dates(rows, None, effective_dates or [])
    latest_date = window_dates[-1] if window_dates else pd.to_datetime(rows["date"]).dt.date.max()
    rows["date"] = pd.to_datetime(rows["date"], errors="coerce").dt.date
    latest_rows = rows[rows["date"] == latest_date]

    if latest_rows.empty:
        return ["TopK: SKIP (no latest day rows)"], []

    slot_lines: List[str] = []
    day_rollup: Dict[int, Dict[str, float]] = {
        k: {"stake": 0.0, "return": 0.0, "pnl": 0.0, "hits": 0.0} for k in (10, 15, 20, 25, 30, 35, 40)
    }

    best_slot_summary: tuple | None = None

    for slot_id, slot_name in SLOT_NAME_MAP.items():
        slot_rows = latest_rows[latest_rows["slot"] == slot_id]
        if slot_rows.empty:
            slot_lines.append(f"{slot_name}: SKIP TopK")
            continue

        actual_vals = slot_rows.dropna(subset=["actual"])
        if actual_vals.empty:
            slot_lines.append(f"{slot_name}: SKIP TopK (no actual)")
            continue

        actual_val = int(actual_vals.iloc[0]["actual"])
        results = _topk_profit(slot_rows, actual_val, unit_stake)
        for k, res in results.items():
            day_rollup[k]["stake"] += res["stake"]
            day_rollup[k]["return"] += res["return"]
            day_rollup[k]["pnl"] += res["pnl"]
            day_rollup[k]["hits"] += 1 if res["hit"] else 0

        best_k, best_res = max(
            results.items(), key=lambda kv: (kv[1]["pnl"], kv[1]["roi"], -kv[0]))
        rank_note = f" r{best_res['rank']}" if best_res["hit"] and best_res["rank"] is not None else ""
        result_note = "hit" if best_res["hit"] else "miss"
        picks = int(round(best_res["stake"] / unit_stake)) if unit_stake else 0
        slot_lines.append(
            f"{slot_name} | Best-K K{best_k:<2} | Picks {picks:>2} | "
            f"Stake {best_res['stake']:.1f} | Return {best_res['return']:.1f} | "
            f"P&L {best_res['pnl']:+.1f} | ROI {best_res['roi']:+.0%} | "
            f"Result {actual_val:02d} ({result_note}{rank_note})"
        )

        roi_component = best_res["roi"] if best_res.get("roi") is not None else -1e9
        candidate_key = (
            best_res["pnl"],
            roi_component,
            -best_k,
        )
        if best_slot_summary is None or candidate_key > best_slot_summary[0]:
            best_slot_summary = (candidate_key, slot_name, best_k, best_res)

    day_lines: List[str] = []
    valid_rollups = {k: v for k, v in day_rollup.items() if v["stake"] > 0}
    if not valid_rollups:
        day_lines.append("Day TopK: SKIP (no data)")
    else:
        best_k, best_vals = max(valid_rollups.items(), key=lambda kv: (kv[1]["pnl"], kv[1]["pnl"] / kv[1]["stake"], -kv[0]))
        roi = best_vals["pnl"] / best_vals["stake"] if best_vals["stake"] else 0.0
        picks = int(round(best_vals["stake"] / unit_stake)) if unit_stake else 0
        summary = (
            f"BEST TOP-K STRATEGY YESTERDAY\n"
            f"Best: K{best_k} → Stake {best_vals['stake']:.0f} | Return {best_vals['return']:.0f} | "
            f"P&L {best_vals['pnl']:+.0f} | ROI {roi:+.0%}"
        )
        if best_slot_summary is not None:
            _, slot_name, win_k, win_res = best_slot_summary
            rank_note = f", rank {win_res['rank']}" if win_res.get("rank") is not None else ""
            summary += f"\nWinning Slot: {slot_name} (K{win_k}, {'HIT' if win_res['hit'] else 'MISS'}{rank_note})"
        day_lines.append(summary)

    return slot_lines, day_lines


def format_topk_profit(
    rows: pd.DataFrame, effective_dates: Optional[List[date]] = None, unit_stake: float = 1.0
) -> Tuple[List[str], List[str]]:
    return _format_topk(rows, effective_dates=effective_dates, unit_stake=unit_stake)


def _cross_slot_matrix(rows: pd.DataFrame) -> List[str]:
    if rows.empty:
        return ["Cross-slot: SKIP no data"]
    end = pd.to_datetime(rows["date"].max())
    start = end - pd.Timedelta(days=59)
    work = rows[(pd.to_datetime(rows["date"]) >= start) & (pd.to_datetime(rows["date"]) <= end)]
    if work.empty:
        return ["Cross-slot: SKIP no data"]
    counter: Dict[Tuple[int, int], int] = defaultdict(int)
    actual_map: Dict[Tuple[date, int], int] = {}
    for _, row in work.dropna(subset=["actual"]).iterrows():
        actual_map[(row["date"], row["slot"])] = int(row["actual"])
    for (day, slot), actual_val in actual_map.items():
        for other_slot in SLOT_NAME_MAP.keys():
            preds = work[(work["date"] == day) & (work["slot"] == other_slot)]
            if preds.empty:
                continue
            hit_rows = preds[preds["number"] == actual_val]
            if not hit_rows.empty:
                counter[(other_slot, slot)] += 1
    pairs = sorted(counter.items(), key=lambda kv: kv[1], reverse=True)
    top = pairs[:2]
    if not top:
        return ["Cross-slot: SKIP no hits"]
    return [f"Cross-slot top: {SLOT_NAME_MAP[a]}→{SLOT_NAME_MAP[b]}={c}" for (a, b), c in top]


DIGIT_HIT_RATE_THRESHOLD = 0.5
POSITIVE_DIGIT_ROI = 0.10


def _digit_roi_by_window(
    slot_digit_hits: pd.DataFrame, window_days: Optional[int], cfg: "PnLConfig", effective_dates: Optional[List[date]] = None
) -> Dict[int, Tuple[float, float]]:
    if slot_digit_hits.empty:
        return {}

    work = slot_digit_hits.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce").dt.date
    window_dates = _select_window_dates(work, window_days, effective_dates or [])
    if window_dates:
        work = work[work["date"].isin(set(window_dates))]
    if work.empty:
        return {}

    work["digit_hit"] = work[["andar_hit", "bahar_hit"]].any(axis=1)
    roi_map: Dict[int, Tuple[float, float]] = {}
    for slot_id, frame in work.groupby("slot"):
        cost = len(frame) * 2 * cfg.cost_per_unit
        payout = frame[["andar_hit", "bahar_hit"]].astype(float).sum().sum() * cfg.digit_payout_per_unit
        pnl = payout - cost
        roi = pnl / cost if cost else 0.0
        hit_rate = frame["digit_hit"].mean() if not frame.empty else 0.0
        roi_map[int(slot_id)] = (roi, hit_rate)
    return roi_map


def format_andar_bahar_gating(
    slot_digit_hits: pd.DataFrame, cfg: "PnLConfig", effective_dates: Optional[List[date]] = None
) -> Tuple[List[str], Dict[str, bool]]:
    header_date: Optional[date] = None
    if not slot_digit_hits.empty and "date" in slot_digit_hits.columns:
        latest_hit_date = slot_digit_hits["date"].dropna().max()
        if isinstance(latest_hit_date, pd.Timestamp):
            header_date = latest_hit_date.date()
        else:
            header_date = latest_hit_date
    if header_date is None and effective_dates:
        header_date = max(effective_dates)

    if slot_digit_hits.empty:
        if effective_dates:
            last_date = max(effective_dates)
            return (
                [f"ANDAR/BAHAR GATE FOR NEXT DAY (based on data up to {last_date}): No digit data"],
                {},
            )
        return (["ANDAR/BAHAR GATE FOR NEXT DAY: No digit data"], {})

    roi_7d = _digit_roi_by_window(slot_digit_hits, 7, cfg, effective_dates)
    roi_30d = _digit_roi_by_window(slot_digit_hits, 30, cfg, effective_dates)
    roi_full = _digit_roi_by_window(slot_digit_hits, None, cfg, effective_dates)

    if effective_dates:
        last_date = max(effective_dates)
        header = f"ANDAR/BAHAR GATE FOR NEXT DAY (based on data up to {last_date})"
    else:
        header = "ANDAR/BAHAR GATE FOR NEXT DAY"

    gate_status: Dict[str, bool] = {}
    lines: List[str] = [header]
    lines.append("Slot  Gate  7D ROI  30D ROI  ALL ROI  Hit7D")
    lines.append("----  ----  -------  -------  -------  -----")
    
    for slot_id, slot_name in SLOT_NAME_MAP.items():
        r7, h7 = roi_7d.get(slot_id, (None, None))
        r30, h30 = roi_30d.get(slot_id, (None, None))
        rfull, _ = roi_full.get(slot_id, (None, None))

        decision = "[SKIP]" if (r7 is None and r30 is None and rfull is None) else "[OFF]"
        if r30 is not None and r30 < 0 and r7 is not None and r7 < 0:
            decision = "[OFF]"
        elif (r30 is not None and r30 > POSITIVE_DIGIT_ROI) or (h7 is not None and h7 >= DIGIT_HIT_RATE_THRESHOLD):
            decision = "[ON]"
        elif rfull is not None and rfull > POSITIVE_DIGIT_ROI:
            decision = "[ON]"

        gate_status[slot_name] = decision == "[ON]"

        def _fmt_roi(val: Optional[float], window_days: Optional[int]) -> str:
            # FIX J: Add insufficient checks
            if window_days is not None:
                sufficient, reason = _check_window_sufficiency(effective_dates, window_days)
                if not sufficient:
                    return f"{reason:>7}"
            if val is None:
                return "   n/a"
            return f"{val:>+6.0%}"

        def _fmt_hit(val: Optional[float]) -> str:
            if val is None:
                return "  n/a"
            return f"{val:>4.0%}"

        lines.append(
            f"{slot_name:<5} {decision:<7} {_fmt_roi(r7, 7)}  {_fmt_roi(r30, 30)}  {_fmt_roi(rfull, None)}  {_fmt_hit(h7)}"
        )

    return lines, gate_status


@dataclass
class PnLConfig:
    cost_per_unit: float = 10.0
    payout_per_unit: float = 900.0  # Example: 90x payout on a correct 2-digit hit
    digit_payout_per_unit: float = 90.0  # Andar/Bahar digit payout (₹1 -> ₹9)
    stake_column: str = "stake"


@dataclass
class PnLReport:
    cleaned_predictions: pd.DataFrame
    merged: pd.DataFrame
    slot_totals: pd.DataFrame
    day_totals: pd.DataFrame
    summary_table: pd.DataFrame
    window_totals: pd.DataFrame
    combined_window_totals: pd.DataFrame
    slot_digit_hits: pd.DataFrame
    digit_pnl: pd.DataFrame
    signal_totals: pd.DataFrame
    hit_notes: pd.DataFrame


# -------------------------
# Helpers
# -------------------------


def _normalize_predictions(df: pd.DataFrame, cfg: PnLConfig) -> pd.DataFrame:
    """Clean and standardize the predictions DataFrame."""

    required_cols = {"date", "slot", "number"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required prediction columns: {sorted(missing)}")

    work = df.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce").dt.date
    work["slot"] = pd.to_numeric(work["slot"], errors="coerce").astype("Int64")
    work["number"] = pd.to_numeric(work["number"], errors="coerce").astype("Int64")

    if cfg.stake_column in work.columns:
        work[cfg.stake_column] = pd.to_numeric(work[cfg.stake_column], errors="coerce")
    else:
        work[cfg.stake_column] = 1.0

    work = work.dropna(subset=["date", "slot", "number"])
    work = work[(work["slot"].isin(SLOT_NAME_MAP.keys())) & (work["number"].between(0, 99))]
    if work.empty:
        print("[WARNING] No valid predictions after cleaning; skipping P&L computation")
        return work

    work[cfg.stake_column] = work[cfg.stake_column].fillna(1.0).clip(lower=0)
    work["number"] = work["number"].astype(int)
    work["slot"] = work["slot"].astype(int)

    # Month-end skip rule: drop last calendar date of every month
    month_end_mask = pd.to_datetime(work["date"]).dt.is_month_end
    if month_end_mask.any():
        skipped_dates = sorted(work.loc[month_end_mask, "date"].unique())
        print(
            "[WARNING] Month-end skip: dropping predictions for "
            + ", ".join(str(d) for d in skipped_dates)
        )
        work = work.loc[~month_end_mask].copy()

    # Optional andar/bahar columns: keep at most one alias each
    def _pick_and_clean_alias(aliases: Iterable[str]) -> Optional[str]:
        present = [c for c in aliases if c in work.columns]
        if not present:
            return None
        col = present[0]
        work[col] = pd.to_numeric(work[col], errors="coerce")
        work[col] = work[col].where(work[col].between(0, 9))
        return col

    andar_col = _pick_and_clean_alias(ANDAR_ALIASES)
    bahar_col = _pick_and_clean_alias(BAHAR_ALIASES)

    if andar_col and andar_col != "andar":
        work = work.rename(columns={andar_col: "andar"})
    elif andar_col is None:
        work = work.drop(columns=[c for c in ANDAR_ALIASES if c in work.columns], errors="ignore")

    if bahar_col and bahar_col != "bahar":
        work = work.rename(columns={bahar_col: "bahar"})
    elif bahar_col is None:
        work = work.drop(columns=[c for c in BAHAR_ALIASES if c in work.columns], errors="ignore")

    return work


def _load_results_long() -> pd.DataFrame:
    """Load real results and convert to long format."""

    df_results = load_results_dataframe()
    if df_results.empty:
        raise ValueError("Results file is empty or missing")

    long_df = df_results.melt(id_vars=["DATE"], var_name="slot_name", value_name="actual")
    long_df = long_df.rename(columns={"DATE": "date"})
    long_df["date"] = pd.to_datetime(long_df["date"], errors="coerce").dt.date
    long_df["slot"] = long_df["slot_name"].map({v: k for k, v in SLOT_NAME_MAP.items()})
    long_df = long_df.dropna(subset=["date", "slot", "actual"])
    long_df["actual"] = pd.to_numeric(long_df["actual"], errors="coerce").astype(int)
    long_df["slot"] = long_df["slot"].astype(int)

    # Month-end skip rule: drop the last calendar date of every month
    month_end_mask = pd.to_datetime(long_df["date"]).dt.is_month_end
    if month_end_mask.any():
        skipped_dates = sorted(long_df.loc[month_end_mask, "date"].unique())
        print(
            "[WARNING] Month-end skip: dropping results for "
            + ", ".join(str(d) for d in skipped_dates)
        )
        long_df = long_df.loc[~month_end_mask].copy()

    return long_df[["date", "slot", "actual"]]



def _mirror_number(value: int) -> int:
    tens, ones = divmod(int(value), 10)
    return ones * 10 + tens


def _is_family_164950_member(value: int) -> bool:
    tens, ones = divmod(int(value), 10)
    return tens in FAMILY_164950_DIGITS and ones in FAMILY_164950_DIGITS


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

    # Always tie-break deterministically on the number itself
    cols.append("number")
    ascending.append(True)

    return cols, ascending


def _derive_slot_digits(cleaned_preds: pd.DataFrame) -> pd.DataFrame:
    """Pick the heaviest number per date/slot and extract its tens/ones digits."""

    # If explicit andar/bahar are present for a row, prefer that row as-is.
    groups = cleaned_preds.groupby(["date", "slot"], sort=False)
    rows: List[Dict] = []

    for (dt, slot), group in groups:
        candidate = group
        explicit_mask = group[[c for c in ("andar", "bahar") if c in group.columns]].notna().any(axis=1)
        if explicit_mask.any():
            candidate = group[explicit_mask]

        sort_cols, ascending = _weight_sort_columns(candidate)
        pick = candidate.sort_values(by=sort_cols, ascending=ascending).iloc[0]

        andar = int(pick["number"]) // 10 if pd.isna(pick.get("andar", None)) else int(pick["andar"])
        bahar = int(pick["number"]) % 10 if pd.isna(pick.get("bahar", None)) else int(pick["bahar"])

        rows.append(
            {
                "date": dt,
                "slot": int(slot),
                "slot_name": SLOT_NAME_MAP.get(int(slot)),
                "pick_number": int(pick["number"]),
                "andar": andar,
                "bahar": bahar,
            }
        )

    return pd.DataFrame(rows)


def _annotate_signal_flags(merged: pd.DataFrame, results_long: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Append hit-adjacent signals and pack/family tags to the merged frame."""

    annotated = merged.copy()
    annotated["near_miss"] = (annotated["number"] - annotated["actual"]).abs() == 1
    annotated["mirror_hit"] = annotated["actual"] == annotated["number"].apply(_mirror_number)
    annotated["s40_member"] = annotated["number"].isin(S40_PACK)
    annotated["family_164950"] = annotated["number"].apply(_is_family_164950_member)

    actual_lookup: Dict[Tuple[pd.Timestamp, int], int] = (
        results_long.set_index(["date", "slot"])["actual"].to_dict()
    )
    per_date_actuals: Dict[pd.Timestamp, List[Tuple[int, int]]] = {
        date: list(zip(group["slot"], group["actual"])) for date, group in results_long.groupby("date")
    }

    def _cross_slot(row: pd.Series) -> bool:
        pairs = per_date_actuals.get(row["date"], [])
        return any(slot != row["slot"] and actual == row["number"] for slot, actual in pairs)

    def _cross_day(row: pd.Series, delta: int) -> bool:
        if row["slot"] == 4:
            return False
        target_date = row["date"] + timedelta(days=delta)
        actual = actual_lookup.get((target_date, row["slot"]))
        return actual == row["number"]

    annotated["cross_slot_hit"] = annotated.apply(_cross_slot, axis=1)
    annotated["cross_day_prev_hit"] = annotated.apply(lambda r: _cross_day(r, -1), axis=1)
    annotated["cross_day_next_hit"] = annotated.apply(lambda r: _cross_day(r, 1), axis=1)
    annotated["cross_day_blocked_by_dswr"] = annotated["slot"] == 4

    signal_rows = [
        ("near_miss", int(annotated["near_miss"].sum())),
        ("mirror_hit", int(annotated["mirror_hit"].sum())),
        ("cross_slot_hit", int(annotated["cross_slot_hit"].sum())),
        ("cross_day_prev_hit", int(annotated["cross_day_prev_hit"].sum())),
        ("cross_day_next_hit", int(annotated["cross_day_next_hit"].sum())),
        ("dswr_boundary_blocks", int(annotated["cross_day_blocked_by_dswr"].sum())),
        ("s40_hits", int((annotated["hit"] & annotated["s40_member"]).sum())),
        ("family_164950_hits", int((annotated["hit"] & annotated["family_164950"]).sum())),
    ]

    return annotated, pd.DataFrame(signal_rows, columns=["metric", "value"])


def _build_hit_notes(merged: pd.DataFrame) -> pd.DataFrame:
    if merged.empty:
        return pd.DataFrame(columns=["date", "slot", "slot_name", "actual", "note"])

    hits = merged[merged["hit"]].copy()
    notes: List[Dict[str, object]] = []

    for _, row in hits.iterrows():
        parts: List[str] = []
        rank = row.get("rank")
        in_top = row.get("in_top")

        if pd.notna(rank):
            parts.append(f"rank {int(rank)}")
        if pd.notna(in_top):
            parts.append(f"in_top={int(in_top)}")
        if row.get("s40_member", False):
            parts.append("S40")
        if row.get("family_164950", False):
            parts.append("164950")
        if row.get("mirror_hit", False):
            parts.append("mirror")
        if row.get("cross_slot_hit", False):
            parts.append("cross-slot")
        if row.get("cross_day_prev_hit", False):
            parts.append("prev-day")
        if row.get("cross_day_next_hit", False):
            parts.append("next-day")

        note = ", ".join(parts)
        notes.append(
            {
                "date": row["date"],
                "slot": row["slot"],
                "slot_name": row.get("slot_name", SLOT_NAME_MAP.get(int(row["slot"]), "")),
                "actual": row["actual"],
                "note": note,
            }
        )

    return pd.DataFrame(notes)


def _build_digit_pnl(
    slot_digit_hits: pd.DataFrame,
    cfg: PnLConfig,
    gate_by_day: Optional[Dict[date, Dict[int, bool]]] = None,
) -> pd.DataFrame:
    """Compute cost/payout for andar/bahar digit bets per slot."""

    if slot_digit_hits.empty:
        return pd.DataFrame(columns=["date", "slot", "slot_name", "cost", "payout", "pnl", "andar_hit", "bahar_hit"])

    frame = slot_digit_hits.copy()
    hit_total = frame[["andar_hit", "bahar_hit"]].astype(float).sum(axis=1)

    if gate_by_day:
        def _gate_on(row: pd.Series) -> bool:
            gate_for_day = gate_by_day.get(row["date"], {})
            return bool(gate_for_day.get(int(row["slot"]), False))

        frame["gate_on"] = frame.apply(_gate_on, axis=1)
        gate_mask = frame["gate_on"].astype(float)
        frame["cost"] = gate_mask * 2 * cfg.cost_per_unit  # fixed ₹20 per slot for AB (₹10 + ₹10)
        frame["payout"] = gate_mask * hit_total * cfg.digit_payout_per_unit
        frame["pnl"] = frame["payout"] - frame["cost"]
    else:
        frame["cost"] = 2 * cfg.cost_per_unit  # fixed ₹20 per slot for AB (₹10 + ₹10)
        frame["payout"] = hit_total * cfg.digit_payout_per_unit
        frame["pnl"] = frame["payout"] - frame["cost"]

    keep_cols = ["date", "slot", "slot_name", "cost", "payout", "pnl", "andar_hit", "bahar_hit"]
    return frame[keep_cols]


def _summarize_windows(merged: pd.DataFrame, effective_dates: Optional[List[date]] = None) -> pd.DataFrame:
    """Compute day/7-day/month-to-date/cumulative aggregates using aligned dates."""

    window_names = ("day", "7d", "month", "cumulative")
    if merged.empty:
        rows = [{"window": name, "stake": 0.0, "payout": 0.0, "pnl": 0.0, "roi": 0.0} for name in window_names]
        return pd.DataFrame(rows)

    available_dates = sorted({d for d in pd.to_datetime(merged["date"], errors="coerce").dt.date.dropna().tolist()})
    aligned_dates = [d for d in (effective_dates or available_dates) if d in set(available_dates)] or available_dates

    latest_date = aligned_dates[-1]
    last_7 = set(aligned_dates[-7:])

    month_mask = merged["date"].apply(lambda d: d.replace(day=1)) == latest_date.replace(day=1)
    windows = {
        "day": merged[merged["date"] == latest_date],
        "7d": merged[merged["date"].isin(last_7)],
        "month": merged[month_mask],
        "cumulative": merged,
    }

    rows = []
    for name in window_names:
        frame = windows.get(name, pd.DataFrame())
        if frame.empty:
            stake = payout = pnl = roi = 0.0
        else:
            stake = frame["cost"].sum()
            payout = frame["payout"].sum()
            pnl = frame["pnl"].sum()
            roi = pnl / stake if stake else 0.0

        rows.append({"window": name, "stake": stake, "payout": payout, "pnl": pnl, "roi": roi})

    return pd.DataFrame(rows)


# -------------------------
# Public API
# -------------------------


def _empty_report(cleaned_preds: pd.DataFrame, cfg: PnLConfig) -> PnLReport:
    base_cols = cleaned_preds.columns.tolist()
    if not base_cols:
        base_cols = ["date", "slot", "number", cfg.stake_column]

    empty_frame = pd.DataFrame(
        columns=base_cols
        + [
            "actual",
            "hit",
            "actual_andar",
            "actual_bahar",
            "cost",
            "payout",
            "pnl",
            "slot_name",
        ]
    )

    return PnLReport(
        cleaned_predictions=cleaned_preds,
        merged=empty_frame,
        slot_totals=pd.DataFrame(columns=["slot", "slot_name", "cost", "payout", "pnl", "roi", "hits", "attempts"]),
        day_totals=pd.DataFrame(columns=["date", "pnl_per_day"]),
        summary_table=pd.DataFrame(columns=["metric", "value"]),
        window_totals=_summarize_windows(pd.DataFrame(columns=["date", "cost", "payout", "pnl"])),
        combined_window_totals=_summarize_windows(pd.DataFrame(columns=["date", "cost", "payout", "pnl"])),
        slot_digit_hits=pd.DataFrame(
            columns=["date", "slot", "slot_name", "andar", "bahar", "andar_hit", "bahar_hit", "hit"]
        ),
        digit_pnl=pd.DataFrame(columns=["date", "slot", "slot_name", "cost", "payout", "pnl", "andar_hit", "bahar_hit"]),
        signal_totals=pd.DataFrame(columns=["metric", "value"]),
        hit_notes=pd.DataFrame(columns=["date", "slot", "slot_name", "actual", "note"]),
    )


def compute_pnl_report(
    predictions: pd.DataFrame,
    cfg: Optional[PnLConfig] = None,
    gate_by_day: Optional[Dict[date, Dict[int, bool]]] = None,
) -> PnLReport:
    """Compute detailed P&L given predictions and real results."""

    cfg = cfg or PnLConfig()
    cleaned_preds = _normalize_predictions(predictions, cfg)

    effective_dates: List[date] = []
    if not cleaned_preds.empty:
        min_date = cleaned_preds["date"].min()
        max_date = cleaned_preds["date"].max()
        effective_dates = build_effective_dates(min_date, max_date, available_dates=_available_result_dates())
        if effective_dates:
            cleaned_preds = cleaned_preds[cleaned_preds["date"].isin(set(effective_dates))]

    if cleaned_preds.empty:
        return _empty_report(cleaned_preds, cfg)

    slot_digits = _derive_slot_digits(cleaned_preds)
    results_long = _load_results_long()
    if effective_dates:
        results_long = results_long[results_long["date"].isin(set(effective_dates))]

    merged = cleaned_preds.merge(results_long, on=["date", "slot"], how="left", indicator=True)

    # Identify predictions that don't yet have a matching result (e.g., office-closed "xx" rows)
    missing_mask = merged["actual"].isna()
    if missing_mask.any():
        missing_dates = sorted(merged.loc[missing_mask, "date"].unique())
        print(
            "[WARNING] Skipping P&L because actual results are missing for: "
            + ", ".join(str(d) for d in missing_dates)
        )

    merged = merged.dropna(subset=["actual"])

    # If everything is pending, return an empty-but-well-formed report instead of raising
    if merged.empty:
        return _empty_report(cleaned_preds, cfg)

    merged["hit"] = merged["number"] == merged["actual"]
    merged["actual_andar"] = merged["actual"] // 10
    merged["actual_bahar"] = merged["actual"] % 10
    merged["cost"] = merged[cfg.stake_column] * cfg.cost_per_unit
    merged["payout"] = merged["hit"].astype(float) * merged[cfg.stake_column] * cfg.payout_per_unit
    merged["pnl"] = merged["payout"] - merged["cost"]
    merged["slot_name"] = merged["slot"].map(SLOT_NAME_MAP)

    merged, signal_totals = _annotate_signal_flags(merged, results_long)
    hit_notes = _build_hit_notes(merged)

    # Slot-level andar/bahar hits based on the heaviest pick
    actual_digits = (
        merged.groupby(["date", "slot", "slot_name"])[["actual_andar", "actual_bahar"]]
        .first()
        .reset_index()
    )
    slot_digit_hits = slot_digits.merge(actual_digits, on=["date", "slot", "slot_name"], how="left")
    slot_digit_hits["andar_hit"] = slot_digit_hits["andar"] == slot_digit_hits["actual_andar"]
    slot_digit_hits["bahar_hit"] = slot_digit_hits["bahar"] == slot_digit_hits["actual_bahar"]
    digit_pnl = _build_digit_pnl(slot_digit_hits, cfg, gate_by_day=gate_by_day)

    slot_totals = (
        merged.groupby(["slot", "slot_name"], dropna=False)[["cost", "payout", "pnl"]]
        .sum()
        .reset_index()
        .sort_values("slot")
    )

    day_totals = merged.groupby("date")["pnl"].sum().reset_index().rename(columns={"pnl": "pnl_per_day"})

    summary_rows = [
        ("bets", len(merged)),
        ("hits", int(merged["hit"].sum())),
        ("hit_rate", merged["hit"].mean()),
        ("cost", merged["cost"].sum()),
        ("payout", merged["payout"].sum()),
        ("net_pnl", merged["pnl"].sum()),
    ]

    hit_rank_series = merged.loc[merged["hit"] & merged["rank"].notna(), "rank"]
    if not hit_rank_series.empty:
        summary_rows.append(("avg_hit_rank", hit_rank_series.mean()))

    summary_rows.extend([(row["metric"], row["value"]) for _, row in signal_totals.iterrows()])

    if not slot_digit_hits.empty:
        total_slots = len(slot_digit_hits)
        andar_hits = int(slot_digit_hits["andar_hit"].sum())
        bahar_hits = int(slot_digit_hits["bahar_hit"].sum())
        summary_rows.extend(
            [
                ("andar_slots", total_slots),
                ("andar_hits", andar_hits),
                ("andar_hit_rate", andar_hits / total_slots if total_slots else 0.0),
                ("bahar_slots", total_slots),
                ("bahar_hits", bahar_hits),
                ("bahar_hit_rate", bahar_hits / total_slots if total_slots else 0.0),
            ]
        )

    summary_table = pd.DataFrame(summary_rows, columns=["metric", "value"])

    window_totals = _summarize_windows(merged, effective_dates)
    combined_frame = pd.concat(
        [merged[["date", "cost", "payout", "pnl"]], digit_pnl[["date", "cost", "payout", "pnl"]]],
        ignore_index=True,
    )
    combined_window_totals = _summarize_windows(combined_frame, effective_dates)

    return PnLReport(
        cleaned_predictions=cleaned_preds,
        merged=merged,
        slot_totals=slot_totals,
        day_totals=day_totals,
        summary_table=summary_table,
        window_totals=window_totals,
        combined_window_totals=combined_window_totals,
        slot_digit_hits=slot_digit_hits,
        digit_pnl=digit_pnl,
        signal_totals=signal_totals,
        hit_notes=hit_notes,
    )


def render_compact_report(report: PnLReport) -> str:
    """Return a human-readable compact string summary."""

    lines: List[str] = []

    lines.append("=== P&L SUMMARY ===")
    for _, row in report.summary_table.iterrows():
        if row["metric"].endswith("rate"):
            value_str = f"{row['value']:.2%}"
        else:
            value_str = f"{row['value']:.2f}" if isinstance(row["value"], float) else str(row["value"])
        lines.append(f"{row['metric']:>12}: {value_str}")

    lines.append("\n=== SLOT TOTALS ===")
    for _, row in report.slot_totals.iterrows():
        lines.append(
            f"{row['slot_name']} | cost {row['cost']:.2f} | payout {row['payout']:.2f} | pnl {row['pnl']:.2f}"
        )

    lines.append("\n=== DAILY PNL ===")
    for _, row in report.day_totals.iterrows():
        lines.append(f"{row['date']}: {row['pnl_per_day']:.2f}")

    lines.append("\n=== WINDOW ROLLOUPS (MAIN) ===")
    for _, row in report.window_totals.iterrows():
        lines.append(
            f"{row['window']:>10}: stake {row['stake']:.2f} | payout {row['payout']:.2f} "
            f"| pnl {row['pnl']:.2f} | roi {row['roi']:.2%}"
        )

    if not report.combined_window_totals.empty:
        lines.append("\n=== WINDOW ROLLOUPS (COMBINED MAIN + ANDAR/BAHAR) ===")
        for _, row in report.combined_window_totals.iterrows():
            lines.append(
                f"{row['window']:>10}: stake {row['stake']:.2f} | payout {row['payout']:.2f} "
                f"| pnl {row['pnl']:.2f} | roi {row['roi']:.2%}"
            )

    if not report.signal_totals.empty:
        lines.append("\n=== SIGNAL COUNTS ===")
        for _, row in report.signal_totals.iterrows():
            lines.append(f"{row['metric']:>18}: {int(row['value'])}")

    if not report.slot_digit_hits.empty:
        lines.append("\n=== ANDAR/BAHAR PICKS (heaviest per slot) ===")
        for _, row in report.slot_digit_hits.sort_values(["date", "slot"]).iterrows():
            lines.append(
                f"{row['date']} {row['slot_name']}: tens {row['andar']} "
                f"{('[HIT]' if row['andar_hit'] else 'miss')} | ones {row['bahar']} "
                f"{('[HIT]' if row['bahar_hit'] else 'miss')}"
            )

    if not report.digit_pnl.empty:
        lines.append("\n=== ANDAR/BAHAR PNL ===")
        for _, row in report.digit_pnl.sort_values(["date", "slot"]).iterrows():
            lines.append(
                f"{row['date']} {row['slot_name']}: cost {row['cost']:.2f} | payout {row['payout']:.2f} | pnl {row['pnl']:.2f}"
            )

    if not report.hit_notes.empty:
        lines.append("\n=== HIT NOTES (rank/top-N + pack/family signals) ===")
        for _, row in report.hit_notes.sort_values(["date", "slot"]).iterrows():
            suffix = f" | {row['note']}" if row["note"] else ""
            lines.append(f"{row['date']} {row['slot_name']}: actual {row['actual']:02d}{suffix}")

    return "\n".join(lines)


def _write_prebuilt_metrics(rows: pd.DataFrame, cfg: PnLConfig, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    start_date = pd.to_datetime(rows["date"], errors="coerce").dt.date.min() if not rows.empty else None
    end_date = pd.to_datetime(rows["date"], errors="coerce").dt.date.max() if not rows.empty else None
    effective_dates = (
        build_effective_dates(start_date, end_date, available_dates=_available_result_dates()) if start_date and end_date else []
    )

    windows = {
        "7d": aggregate_metrics(rows, window_days=7, effective_dates=effective_dates),
        "30d": aggregate_metrics(rows, window_days=30, effective_dates=effective_dates),
        "full": aggregate_metrics(rows, window_days=None, effective_dates=effective_dates),
    }

    for name, df in windows.items():
        out_path = output_dir / f"prebuilt_hero_weak_{name}.csv"
        df.to_csv(out_path, index=False)

    info = {
        "start": str(start_date) if start_date else None,
        "end": str(end_date) if end_date else None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "unit_stake": cfg.cost_per_unit,
    }
    (output_dir / "build_info.json").write_text(json.dumps(info, indent=2))


def _load_build_info(prebuilt_dir: Path) -> Dict[str, object]:
    info_path = prebuilt_dir / "build_info.json"
    if not info_path.exists():
        return {}
    try:
        return json.loads(info_path.read_text())
    except json.JSONDecodeError:
        return {}


def prebuilt_metrics_status(latest_result_date: Optional[date], prebuilt_dir: Path = Path(PREBUILT_DIR)) -> Tuple[bool, str]:
    prebuilt_dir = Path(prebuilt_dir)
    required = [prebuilt_dir / f"prebuilt_hero_weak_{name}.csv" for name in ("7d", "30d", "full")]
    if any(not p.exists() for p in required):
        return True, "missing prebuilt metrics"

    info = _load_build_info(prebuilt_dir)
    end_str = info.get("end") if isinstance(info, dict) else None
    if latest_result_date and end_str:
        try:
            build_end = date.fromisoformat(end_str)
            if build_end < latest_result_date:
                return True, "prebuilt older than latest results"
        except ValueError:
            return True, "build_info unreadable"

    return False, ""


def load_prebuilt_metrics(prebuilt_dir: Path = Path(PREBUILT_DIR)) -> Dict[str, pd.DataFrame]:
    prebuilt_dir = Path(prebuilt_dir)
    metrics: Dict[str, pd.DataFrame] = {}
    for name in ("7d", "30d", "full"):
        path = prebuilt_dir / f"prebuilt_hero_weak_{name}.csv"
        if path.exists():
            metrics[name] = pd.read_csv(path)
    return metrics


def rebuild_prebuilt_metrics(start_date: date, end_date: date, cfg: PnLConfig) -> Dict[str, pd.DataFrame]:
    rows = load_clean_bet_rows(start_date, end_date, cfg)
    if rows.empty:
        _log_skip("no rows to build prebuilt metrics")
        return {}

    output_dir = Path(PREBUILT_DIR)
    _write_prebuilt_metrics(rows, cfg, output_dir)
    return load_prebuilt_metrics(output_dir)


def _cli_rebuild_all():
    import argparse

    parser = argparse.ArgumentParser(description="Rebuild hero/weak prebuilt metrics")
    parser.add_argument("--rebuild", choices=["all"], required=False)
    args = parser.parse_args()

    if args.rebuild != "all":
        parser.error("Use --rebuild all to rebuild metrics")

    start_input = input("Start date (DD-mm-yy): ").strip()
    end_input = input("End date (blank = last result date): ").strip()

    results_df = load_results_dataframe()
    if results_df.empty:
        _log_skip("results file missing or empty")
        return

    results_df["DATE"] = pd.to_datetime(results_df["DATE"], errors="coerce").dt.date
    last_date = results_df["DATE"].max()

    try:
        start_date = datetime.strptime(start_input, "%d-%m-%y").date() if start_input else last_date
        end_date = datetime.strptime(end_input, "%d-%m-%y").date() if end_input else last_date
    except ValueError:
        _log_skip("invalid date format")
        return

    cfg = PnLConfig()
    metrics = rebuild_prebuilt_metrics(start_date, end_date, cfg)
    if not metrics:
        _log_skip("no rows to build prebuilt metrics")
        return

    print(f"Prebuilt metrics written to {PREBUILT_DIR}")


__all__ = [
    "PnLConfig",
    "PnLReport",
    "compute_pnl_report",
    "render_compact_report",
    "build_effective_dates",
    "load_clean_bet_rows",
    "aggregate_metrics",
    "format_hero_weakest",
    "format_topk_profit",
    "format_rank_bucket_windows",
    "format_tag_roi",
    "format_cross_slot_hits",
    "format_andar_bahar_gating",
    "load_prebuilt_metrics",
    "prebuilt_metrics_status",
    "rebuild_prebuilt_metrics",
]


if __name__ == "__main__":
    _cli_rebuild_all()
