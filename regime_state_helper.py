"""Lightweight regime tracking for family packs using recent ROI/drawdown.

This helper stays intentionally small and only relies on the existing
``quant_data_core`` loader plus shared pack definitions from
``bet_pnl_tracker``. It computes per-slot OFF/NORMAL/BOOST states so SCR9 can
adjust vote weights without inflating the codebase.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Callable, Dict, Iterable, List

import pandas as pd

from bet_pnl_tracker import (
    FAMILY_164950_DIGITS,
    PnLConfig,
    S40_PACK,
    SLOT_NAME_MAP,
    _digit_roi_by_window,
    build_effective_dates,
    compute_pnl_report,
    format_andar_bahar_gating,
    _read_prediction_history,
)
from quant_data_core import load_results_dataframe


@dataclass
class RegimeState:
    state: str
    roi_30d: float | None
    roi_90d: float | None
    drawdown_90d: float | None
    sigma_90d: float | None
    sample_size: int


STATE_MULTIPLIER = {"OFF": 0.6, "NORMAL": 1.0, "BOOST": 1.35}


def _slice_tail(values: List[float], window: int | None) -> List[float]:
    if window is None or len(values) <= window:
        return values
    return values[-window:]


def _daily_pnl(series: Iterable, member_fn: Callable[[int], bool]) -> List[float]:
    pnl: List[float] = []
    for value in series:
        if pd.isna(value):
            continue
        try:
            num = int(value) % 100
        except (TypeError, ValueError):
            continue
        payout = 90.0 if member_fn(num) else 0.0
        pnl.append(payout - 1.0)
    return pnl


def _roi(pnls: List[float]) -> float | None:
    if not pnls:
        return None
    stake = float(len(pnls))
    return sum(pnls) / stake


def _max_drawdown(pnls: List[float]) -> float | None:
    if not pnls:
        return None
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for change in pnls:
        equity += change
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)
    return max_dd


def _sigma(pnls: List[float]) -> float | None:
    if len(pnls) < 2:
        return None
    return float(pd.Series(pnls).std(ddof=0))


def _classify(roi_30d: float | None, drawdown_90d: float | None, sigma_90d: float | None) -> str:
    if roi_30d is not None and drawdown_90d is not None and sigma_90d is not None:
        if roi_30d < -0.10 and drawdown_90d > 1.5 * sigma_90d:
            return "OFF"
    if roi_30d is not None and roi_30d > 0.10:
        return "BOOST"
    return "NORMAL"


def _is_s40(num: int) -> bool:
    return num in S40_PACK


def _is_164950(num: int) -> bool:
    tens, ones = divmod(num, 10)
    return tens in FAMILY_164950_DIGITS and ones in FAMILY_164950_DIGITS


def family_tags_for_number(num: int) -> List[str]:
    tags: List[str] = []
    if _is_s40(num):
        tags.append("S40")
    if _is_164950(num):
        tags.append("164950")
    if len(tags) == 2:
        tags.append("BOTH")
    return tags


def format_regime_note(slot_states: Dict[str, RegimeState]) -> str:
    parts: List[str] = []
    for family in ("S40", "164950", "BOTH"):
        st = slot_states.get(family)
        if not st:
            continue
        roi_text = f"{st.roi_30d:+.0%}" if st.roi_30d is not None else "n/a"
        parts.append(f"{family}:{st.state}({roi_text})")
    return " | ".join(parts) if parts else "n/a"


def compute_regime_states() -> Dict[str, Dict[str, RegimeState]]:
    df = load_results_dataframe()
    if df.empty:
        return {slot: {fam: RegimeState("NORMAL", None, None, None, None, 0) for fam in ("S40", "164950", "BOTH")} for slot in SLOT_NAME_MAP.values()}

    df = df.sort_values("DATE")
    families: Dict[str, Callable[[int], bool]] = {
        "S40": _is_s40,
        "164950": _is_164950,
        "BOTH": lambda n: _is_s40(n) and _is_164950(n),
    }

    states: Dict[str, Dict[str, RegimeState]] = {}
    for slot_name in SLOT_NAME_MAP.values():
        series = df.get(slot_name, pd.Series(dtype=float))
        slot_states: Dict[str, RegimeState] = {}
        for family, member_fn in families.items():
            pnls = _daily_pnl(series, member_fn)
            pnls_90 = _slice_tail(pnls, 90)
            roi_30d = _roi(_slice_tail(pnls, 30))
            roi_90d = _roi(pnls_90)
            dd_90 = _max_drawdown(pnls_90)
            sigma_90 = _sigma(pnls_90)
            state = _classify(roi_30d, dd_90, sigma_90)
            slot_states[family] = RegimeState(
                state=state,
                roi_30d=roi_30d,
                roi_90d=roi_90d,
                drawdown_90d=dd_90,
                sigma_90d=sigma_90,
                sample_size=len(pnls),
            )
        states[slot_name] = slot_states

    return states


def _build_ab_gate_inputs(
    cutoff_dt: date,
) -> tuple[pd.DataFrame, PnLConfig, List[date]]:
    history = _read_prediction_history()
    if history.empty or "date" not in history.columns:
        return pd.DataFrame(), PnLConfig(), []

    history = history[history["date"].notna() & (history["date"] <= cutoff_dt)]
    if history.empty:
        return pd.DataFrame(), PnLConfig(), []

    cfg = PnLConfig()
    report = compute_pnl_report(history, cfg)
    if report.slot_digit_hits.empty:
        return report.slot_digit_hits, cfg, []

    results_df = load_results_dataframe()
    if not results_df.empty:
        results_df["DATE"] = pd.to_datetime(results_df["DATE"], errors="coerce").dt.date
        available_dates = [d for d in results_df["DATE"].dropna().tolist() if d <= cutoff_dt]
    else:
        available_dates = []

    min_date = history["date"].min()
    effective_dates = build_effective_dates(min_date, cutoff_dt, available_dates=available_dates)
    return report.slot_digit_hits, cfg, effective_dates


def compute_ab_gate_snapshot(cutoff_dt: date) -> Dict[int, bool]:
    """
    cutoff_dt inclusive tak ki history use karo (results/predictions strictly <= cutoff_dt).
    Month-end skip rules apply.
    Return: {1: bool, 2: bool, 3: bool, 4: bool}  # FRBD..DSWR
    """
    slot_digit_hits, cfg, effective_dates = _build_ab_gate_inputs(cutoff_dt)
    if slot_digit_hits.empty:
        return {slot_id: False for slot_id in SLOT_NAME_MAP.keys()}

    _, gate_status = format_andar_bahar_gating(slot_digit_hits, cfg, effective_dates)
    return {slot_id: bool(gate_status.get(slot_name, False)) for slot_id, slot_name in SLOT_NAME_MAP.items()}


def compute_ab_gate_snapshot_metrics(
    cutoff_dt: date,
) -> tuple[Dict[int, bool], Dict[int, Dict[str, float | None]]]:
    slot_digit_hits, cfg, effective_dates = _build_ab_gate_inputs(cutoff_dt)
    if slot_digit_hits.empty:
        empty_gate = {slot_id: False for slot_id in SLOT_NAME_MAP.keys()}
        empty_roi = {
            slot_id: {"7d": None, "30d": None, "all": None} for slot_id in SLOT_NAME_MAP.keys()
        }
        return empty_gate, empty_roi

    roi_7d = _digit_roi_by_window(slot_digit_hits, 7, cfg, effective_dates)
    roi_30d = _digit_roi_by_window(slot_digit_hits, 30, cfg, effective_dates)
    roi_all = _digit_roi_by_window(slot_digit_hits, None, cfg, effective_dates)

    _, gate_status = format_andar_bahar_gating(slot_digit_hits, cfg, effective_dates)
    gate_map = {slot_id: bool(gate_status.get(slot_name, False)) for slot_id, slot_name in SLOT_NAME_MAP.items()}

    roi_map: Dict[int, Dict[str, float | None]] = {}
    for slot_id in SLOT_NAME_MAP.keys():
        roi_map[slot_id] = {
            "7d": roi_7d.get(slot_id, (None, None))[0],
            "30d": roi_30d.get(slot_id, (None, None))[0],
            "all": roi_all.get(slot_id, (None, None))[0],
        }
    return gate_map, roi_map


def get_ab_gate_for_day(day: date) -> Dict[int, bool]:
    """Return Andar/Bahar gate status per slot for the given day (strict cutoff)."""
    return compute_ab_gate_snapshot(day - timedelta(days=1))
