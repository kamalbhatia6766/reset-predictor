"""
Experimental SCR9 aggregator that runs SCR1-SCR8, collects candidate numbers,
union-dedupes per slot, and reports vote counts.
"""
import argparse
import glob
import importlib
import inspect
import os
import sys
import traceback
import types
from collections import defaultdict
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd

from bet_pnl_tracker import FAMILY_164950_DIGITS, S40_PACK
from regime_state_helper import (
    STATE_MULTIPLIER,
    compute_regime_states,
    family_tags_for_number,
    format_regime_note,
)
from quant_data_core import compute_learning_signals


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

VERBOSE = False
LOG_PATH = os.path.join("logs", "run_scr9.log")


def log(msg: str, level: str = "INFO", to_console: bool = False):
    """Write to log file and optionally console when verbose."""

    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    line = f"[{level}] {msg}"

    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")

    if to_console:
        print(line)
    elif level == "DEBUG" and VERBOSE:
        print(line)


def log_debug(msg: str):
    log(msg, level="DEBUG")


# ---------------------------------------------------------------------------
# Loader helpers
# ---------------------------------------------------------------------------

def _ensure_loader_modules():
    """Ensure quant_excel_loader/quant_data_core imports work.

    The existing SCR scripts expect these helper modules. In this environment we
    create lightweight fallbacks if the modules are missing so that the scripts
    can run without modification.
    """

    if "quant_excel_loader" not in sys.modules:
        try:
            importlib.import_module("quant_excel_loader")
        except ImportError:
            fallback = types.ModuleType("quant_excel_loader")

            def load_results_excel(path="number prediction learn.xlsx"):
                return pd.read_excel(path)

            fallback.load_results_excel = load_results_excel
            sys.modules["quant_excel_loader"] = fallback

    if "quant_data_core" not in sys.modules:
        try:
            importlib.import_module("quant_data_core")
        except ImportError:
            fallback = types.ModuleType("quant_data_core")

            def load_results_dataframe(path="number prediction learn.xlsx"):
                return pd.read_excel(path)

            fallback.load_results_dataframe = load_results_dataframe
            sys.modules["quant_data_core"] = fallback


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def load_base_dataframe(path: str = "number prediction learn.xlsx") -> pd.DataFrame:
    """Load the Excel using the shared loader utilities if available."""

    _ensure_loader_modules()

    # Prefer quant_data_core if present
    qdc = importlib.import_module("quant_data_core")
    if hasattr(qdc, "load_results_dataframe"):
        loader = qdc.load_results_dataframe
        sig = inspect.signature(loader)
        if len(sig.parameters) == 0:
            return loader()
        return loader(path)

    qel = importlib.import_module("quant_excel_loader")
    if hasattr(qel, "load_results_excel"):
        return qel.load_results_excel(path)

    return pd.read_excel(path)


def to_long_format(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize to [date, slot, number] with numeric slots."""

    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    if "date" not in df.columns and "DATE" in df.columns:
        df["date"] = pd.to_datetime(df["DATE"], errors="coerce")

    slot_map = {"FRBD": 1, "GZBD": 2, "GALI": 3, "DSWR": 4}

    if "slot" in df.columns and "number" in df.columns:
        df["slot"] = df["slot"].apply(lambda x: slot_map.get(x, x))
        df["number"] = pd.to_numeric(df["number"], errors="coerce")
        df = df.dropna(subset=["date", "slot", "number"])
        df["slot"] = df["slot"].astype(int)
        df["number"] = df["number"].astype(int) % 100
        return df[["date", "slot", "number"]]

    slot_cols = [c for c in slot_map if c in df.columns]
    parts = []
    for col in slot_cols:
        part = df[["date", col]].copy()
        part = part.rename(columns={col: "number"})
        part["slot"] = slot_map[col]
        parts.append(part)

    if not parts:
        raise ValueError("Input data missing slot columns FRBD/GZBD/GALI/DSWR")

    long_df = pd.concat(parts, ignore_index=True)
    long_df["number"] = pd.to_numeric(long_df["number"], errors="coerce")
    long_df = long_df.dropna(subset=["date", "slot", "number"])
    long_df["slot"] = long_df["slot"].astype(int)
    long_df["number"] = long_df["number"].astype(int) % 100
    return long_df[["date", "slot", "number"]]


def to_2d(value) -> str:
    try:
        return f"{int(value) % 100:02d}"
    except Exception:
        return None


def get_latest_file(glob_pattern: str) -> Optional[Path]:
    """Return the newest matching file for the glob pattern."""

    matches = [Path(p) for p in glob.glob(glob_pattern)]
    if not matches:
        return None

    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0]


def latest_glob(patterns: Sequence[str], base: Path) -> Optional[Path]:
    """Return the newest file matching any of the provided glob patterns."""

    newest: Optional[Path] = None
    for pattern in patterns:
        candidate = get_latest_file(str(base / pattern))
        if candidate and (newest is None or candidate.stat().st_mtime > newest.stat().st_mtime):
            newest = candidate
    return newest


def _parse_number_cell(value) -> List[str]:
    if pd.isna(value):
        return []

    if isinstance(value, (list, tuple, set)):
        values = value
    elif isinstance(value, str):
        # Split by commas or whitespace
        parts = []
        for chunk in value.replace("/", " ").replace("|", " ").split():
            parts.extend([p.strip() for p in chunk.split(",") if p.strip()])
        values = parts
    else:
        values = [value]

    normalized = [to_2d(v) for v in values]
    return [n for n in normalized if n is not None and len(n) == 2]


def _dedupe_preserve(values: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for value in values:
        if value is None:
            continue
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def read_slot_predictions_xlsx(path: Path) -> Dict[str, List[str]]:
    """Extract slot->numbers from a wide or long Excel prediction file."""

    df = pd.read_excel(path)
    slot_names = {"FRBD", "GZBD", "GALI", "DSWR"}
    result = {slot: [] for slot in slot_names}

    df.columns = [str(c).strip() for c in df.columns]

    # Long format with slot column
    long_slot_cols = {"slot", "slot_name", "Slot", "SlotName"}
    num_cols = {"number", "num", "Number", "predicted_number"}
    slot_col = next((c for c in long_slot_cols if c in df.columns), None)
    num_col = next((c for c in num_cols if c in df.columns), None)

    if slot_col and num_col:
        for slot in slot_names:
            subset = df[df[slot_col].astype(str).str.upper() == slot]
            nums: List[str] = []
            for value in subset[num_col].tolist():
                nums.extend(_parse_number_cell(value))
            result[slot] = nums
        return result

    # Wide format per-slot columns
    wide_slots = [c for c in df.columns if str(c).upper() in slot_names]
    if wide_slots:
        for col in wide_slots:
            slot = str(col).upper()
            values: List[str] = []
            for value in df[col].tolist():
                values.extend(_parse_number_cell(value))
            result[slot] = values
        return result

    # Fallback: treat first four columns as slots if unnamed
    unnamed = [c for c in df.columns if str(c).startswith("Unnamed")]
    if len(unnamed) >= 4:
        for slot, col in zip(sorted(slot_names), unnamed[:4]):
            values: List[str] = []
            for value in df[col].tolist():
                values.extend(_parse_number_cell(value))
            result[slot] = values
        return result

    return result


def _pick_latest_date(df: pd.DataFrame, target_date: Optional[pd.Timestamp]) -> Optional[pd.Timestamp]:
    """Choose the most relevant date from prediction files.

    Preference order:
    1) Explicit target_date if provided and present in the file.
    2) Latest available date in the file.
    """

    date_col = None
    for candidate in ["date", "Date", "DATE"]:
        if candidate in df.columns:
            date_col = candidate
            break

    if date_col is None:
        return None

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    if target_date is not None:
        matches = df[date_col] == pd.to_datetime(target_date)
        if matches.any():
            return target_date

    if df[date_col].notna().any():
        return df[date_col].max()

    return None


def _maybe_limit(values: List[str], limit: Optional[int]) -> List[str]:
    if limit is None:
        return values
    return values[:limit]


def read_predictions_any(
    path: Path,
    *,
    target_date: Optional[pd.Timestamp] = None,
    per_slot_limit: Optional[int] = 30,
) -> Dict[str, List[str]]:
    """Robust reader that normalizes different prediction layouts."""

    df = pd.read_excel(path)
    df.columns = [str(c).strip() for c in df.columns]
    slot_order = ["FRBD", "GZBD", "GALI", "DSWR"]
    slot_ids = {"FRBD": 1, "GZBD": 2, "GALI": 3, "DSWR": 4}
    result = {slot: [] for slot in slot_order}

    latest_date = _pick_latest_date(df, target_date)
    if latest_date is not None:
        date_col = next((c for c in ["date", "Date", "DATE"] if c in df.columns), None)
        df = df[df[date_col] == pd.to_datetime(latest_date)]

    # Direct wide columns (FRBD/GZBD/GALI/DSWR)
    wide_cols = [c for c in df.columns if str(c).upper() in slot_ids]
    if wide_cols:
        for col in wide_cols:
            slot = str(col).upper()
            values: List[str] = []
            for value in df[col].tolist():
                values.extend(_parse_number_cell(value))
            result[slot] = _maybe_limit(_dedupe_preserve(values), per_slot_limit)
        return result

    # Long format: slot + number columns
    slot_candidates = ["slot", "slot_id", "slot_idx", "Slot", "slot_name"]
    num_candidates = ["number", "num", "predicted_number", "Number"]
    slot_col = next((c for c in slot_candidates if c in df.columns), None)
    num_col = next((c for c in num_candidates if c in df.columns), None)

    if slot_col and num_col:
        for slot in slot_order:
            mask = None
            if df[slot_col].dtype.kind in "if":
                mask = df[slot_col] == slot_ids[slot]
            else:
                mask = df[slot_col].astype(str).str.upper() == slot
            subset = df[mask]
            values = [_parse_number_cell(v) for v in subset[num_col].tolist()]
            flattened: List[str] = []
            for chunk in values:
                flattened.extend(chunk)
            result[slot] = _maybe_limit(_dedupe_preserve(flattened), per_slot_limit)
        return result

    # Fallback: assume first four columns map to the slot order
    if df.shape[1] >= 4:
        for slot, col in zip(slot_order, df.columns[:4]):
            values: List[str] = []
            for value in df[col].tolist():
                values.extend(_parse_number_cell(value))
            result[slot] = _maybe_limit(_dedupe_preserve(values), per_slot_limit)
        return result

    return result


# ---------------------------------------------------------------------------
# Prediction adapters
# ---------------------------------------------------------------------------

def _extract_candidates(pred_df: pd.DataFrame, slot_names: Dict[int, str]) -> Dict[str, List[str]]:
    """Standardize prediction dataframe into slot->list of 2-digit strings."""

    result = {name: [] for name in slot_names.values()}
    if pred_df is None or len(pred_df) == 0:
        return result

    df = pred_df.copy()
    slot_col = None
    for candidate in ["slot", "slot_id", "slot_idx", "slot_index"]:
        if candidate in df.columns:
            slot_col = candidate
            break
    if slot_col is None and "slot_name" in df.columns:
        slot_col = "slot_name"
    if slot_col is None:
        return result

    num_col = None
    for candidate in ["number", "num", "predicted_number"]:
        if candidate in df.columns:
            num_col = candidate
            break
    if num_col is None:
        return result

    # Use earliest prediction date to align with "next" draw
    if "date" in df.columns:
        first_date = df["date"].min()
        df = df[df["date"] == first_date]

    for slot_id, slot_name in slot_names.items():
        if slot_col == "slot_name":
            subset = df[df[slot_col] == slot_name]
        else:
            subset = df[df[slot_col] == slot_id]
        nums = [to_2d(n) for n in subset[num_col].tolist()]
        result[slot_name] = [n for n in nums if n is not None]

    return result


def _select_method(predictor):
    """Select the first available prediction method based on priority."""

    candidates = [
        "generate_predictions",
        "predict",
        "run",
        "generate",
        "get_predictions",
        "generate_hybrid_predictions",
        "generate_final_predictions",
    ]

    for name in candidates:
        if hasattr(predictor, name):
            return name
    raise AttributeError("No suitable prediction method found on predictor")


def _prepare_call(method, df_long: pd.DataFrame):
    """Prepare args/kwargs based on the callable signature."""

    sig = inspect.signature(method)
    params = list(sig.parameters.values())
    args = []
    kwargs = {}

    # Pass dataframe to the first positional parameter (excluding self)
    for p in params:
        if p.name == "self":
            continue
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
            args.append(df_long)
            break
        if p.kind == inspect.Parameter.VAR_POSITIONAL:
            args.append(df_long)
            break

    if "days" in sig.parameters:
        kwargs["days"] = 1
    if "top_k" in sig.parameters:
        kwargs["top_k"] = 10

    return args, kwargs, sig


def _run_predictor(module_name: str, class_name: str, df_long: pd.DataFrame, key: str):
    module = importlib.import_module(module_name)
    predictor_cls = getattr(module, class_name)
    predictor = predictor_cls()

    method_name = _select_method(predictor)
    method = getattr(predictor, method_name)
    args, kwargs, sig = _prepare_call(method, df_long)

    if "top_k" not in sig.parameters and "top_k" in kwargs:
        kwargs.pop("top_k", None)

    if key in {"scr3", "scr4"}:
        log(f"{key} using method '{method_name}'", level="DEBUG")

    with open(LOG_PATH, "a", encoding="utf-8") as log_file:
        with redirect_stdout(log_file), redirect_stderr(log_file):
            pred_df = method(*args, **kwargs)

    slot_names = getattr(predictor, "slot_names", {1: "FRBD", 2: "GZBD", 3: "GALI", 4: "DSWR"})
    return _extract_candidates(pred_df, slot_names)


ADAPTERS = {
    "scr1": lambda df: _run_predictor("deepseek_scr1", "PreciseNumberPredictor", df, "scr1"),
    "scr2": lambda df: _run_predictor("deepseek_scr2", "UltimateNumberPredictor", df, "scr2"),
    "scr3": lambda df: _run_predictor("deepseek_scr3", "HybridPredictor", df, "scr3"),
    "scr4": lambda df: _run_predictor("deepseek_scr4", "FinalPredictor", df, "scr4"),
    "scr5": lambda df: _run_predictor("deepseek_scr5", "AutoUpdatePredictor", df, "scr5"),
    "scr6": lambda df: _run_predictor("deepseek_scr6", "UltimatePredictorPro", df, "scr6"),
    "scr7": lambda df: _run_predictor("deepseek_scr7", "AdvancedLearningPredictor", df, "scr7"),
    "scr8": lambda df: _run_predictor("deepseek_scr8", "SCR10UltimatePredictor", df, "scr8"),
}

SCRIPT_PATTERNS: Dict[str, Sequence[str]] = {
    "scr1": ["predictions/precise_predictions.xlsx", "predictions/detailed_predictions.xlsx"],
    "scr2": ["predictions/deepseek_scr2/scr2_predictions_*.xlsx"],
    "scr3": ["predictions/deepseek_scr3/scr3_predictions_*.xlsx"],
    "scr4": ["predictions/deepseek_scr4/scr4_predictions_*.xlsx"],
    "scr5": ["predictions/deepseek_scr5/scr5_predictions_*.xlsx"],
    # SCR6 historically emitted "ultimate_predictions_*" but the latest cache
    # writer saves as scr6_predictions_latest(_detailed).xlsx. Support both to
    # avoid slot dropouts when the new file naming is used.
    "scr6": [
        "predictions/deepseek_scr6/ultimate_predictions_*.xlsx",
        "predictions/deepseek_scr6/scr6_predictions_latest*.xlsx",
    ],
    "scr7": ["predictions/deepseek_scr7/advanced_predictions_*.xlsx"],
    "scr8": ["predictions/deepseek_scr8/scr10_predictions_*.xlsx"],
}


# ---------------------------------------------------------------------------
# Aggregation + regime weighting
# ---------------------------------------------------------------------------


def _regime_multiplier(slot: str, num: str, regime_states: Optional[Dict[str, Dict[str, object]]]) -> float:
    if not regime_states:
        return 1.0

    slot_states = regime_states.get(slot, {})
    try:
        families = family_tags_for_number(int(num))
    except (TypeError, ValueError):
        families = []

    factors: List[float] = []
    for fam in families:
        state_obj = slot_states.get(fam)
        if not state_obj:
            continue
        factors.append(STATE_MULTIPLIER.get(getattr(state_obj, "state", "NORMAL"), 1.0))

    if not factors:
        return 1.0

    multiplier = 1.0
    for f in factors:
        multiplier *= f

    # Keep weights bounded to avoid extreme dominance or collapse
    return min(max(multiplier, 0.25), 3.0)


def _is_flat_board(candidates: List[Dict[str, object]], top_score: float) -> bool:
    if not candidates or top_score <= 0:
        return False

    min_score = min(c["score"] for c in candidates)
    return (top_score - min_score) <= 0.01 * top_score


def _enforce_tens_bin_balance(
    shortlisted: List[Dict[str, object]],
    remainder: List[Dict[str, object]],
    *,
    limit: int,
    min_bins: int = 3,
    max_per_bin: int = 6,
) -> List[Dict[str, object]]:
    counts = defaultdict(int)
    balanced: List[Dict[str, object]] = []

    for row in shortlisted:
        bin_id = int(row["number"]) // 10
        if counts[bin_id] >= max_per_bin:
            remainder.append(row)
            continue
        counts[bin_id] += 1
        balanced.append(row)

    def _bin_count() -> int:
        return sum(1 for v in counts.values() if v > 0)

    if _bin_count() >= min_bins:
        return balanced[:limit]

    for row in list(remainder):
        bin_id = int(row["number"]) // 10
        if counts.get(bin_id, 0) > 0:
            continue

        if len(balanced) >= limit:
            heavy_bin, _ = max(counts.items(), key=lambda kv: (kv[1], kv[0])) if counts else (None, None)
            if heavy_bin is None:
                break
            for idx in range(len(balanced) - 1, -1, -1):
                if int(balanced[idx]["number"]) // 10 == heavy_bin:
                    removed = balanced.pop(idx)
                    counts[heavy_bin] -= 1
                    remainder.append(removed)
                    break

        if len(balanced) < limit:
            balanced.append(row)
            counts[bin_id] += 1

        if _bin_count() >= min_bins:
            break

    return balanced[:limit]


def _apply_selection_rules(entries: List[Dict[str, object]], base_limit: int = 15) -> List[Dict[str, object]]:
    if not entries:
        return []

    top_score = entries[0]["score"]
    threshold = top_score * 0.97
    gated = [row for row in entries if row["score"] >= threshold]

    cap = 12
    if _is_flat_board(gated, top_score):
        cap = 15

    cap = min(cap, int(base_limit)) if base_limit else cap
    shortlisted = gated[:cap]
    remainder = gated[cap:]

    return _enforce_tens_bin_balance(shortlisted, remainder, limit=cap)


def _family_members(family: str) -> set[int]:
    if family == "S40":
        return set(S40_PACK)
    if family == "164950":
        return {10 * a + b for a in FAMILY_164950_DIGITS for b in FAMILY_164950_DIGITS}
    return set()


def _avg_score(rows: List[Dict[str, object]]) -> float:
    if not rows:
        return 0.0
    return float(sum(r["score"] for r in rows)) / len(rows)


def _apply_overlays(
    summary: Dict[str, List[Dict[str, object]]],
    shortlist_map: Dict[str, List[Dict[str, object]]],
    regime_states: Optional[Dict[str, Dict[str, object]]],
    *,
    max_additions: int = 6,
):
    """Conditionally add family overlays without mutating fixed packs."""

    updated_shortlists: Dict[str, List[Dict[str, object]]] = {
        slot: list(rows) for slot, rows in shortlist_map.items()
    }
    overlay_report = {}

    for slot, entries in summary.items():
        base_rows = list(shortlist_map.get(slot, []))
        base_numbers = {row["number"] for row in base_rows}
        base_avg = _avg_score(base_rows)
        remaining = max_additions
        added: List[Dict[str, object]] = []
        status_parts: List[str] = []

        slot_states = regime_states.get(slot, {}) if regime_states else {}
        for family in ("S40", "164950"):
            state = slot_states.get(family)
            roi_30d = getattr(state, "roi_30d", None)
            if roi_30d is None or roi_30d <= 0.10:
                status_parts.append(f"{family}:OFF")
                continue

            family_set = _family_members(family)
            candidates = [
                row
                for row in entries
                if int(row["number"]) in family_set and row["number"] not in base_numbers
            ]
            if not candidates:
                status_parts.append(f"{family}:NONE")
                continue

            selected = candidates[:remaining]
            overlay_avg = _avg_score(selected)
            delta_ev = overlay_avg - base_avg
            if delta_ev <= 0:
                status_parts.append(f"{family}:SKIPΔEV")
                continue

            for row in selected:
                added.append({**row, "overlay_family": family})
            status_parts.append(f"{family}:ON+{len(selected)}(ΔEV={delta_ev:+.2f})")
            remaining -= len(selected)
            if remaining <= 0:
                break

        if added:
            updated_shortlists[slot] = base_rows + added
        overlay_report[slot] = {
            "added": added,
            "status": " | ".join(status_parts) if status_parts else "n/a",
        }

    return updated_shortlists, overlay_report


def aggregate_candidates(all_candidates: Dict[str, Dict[str, List[str]]], regime_states: Optional[Dict[str, Dict[str, object]]]):
    slots = ["FRBD", "GZBD", "GALI", "DSWR"]
    aggregated = {
        slot: defaultdict(lambda: {"votes": 0, "sources": set(), "score": 0.0})
        for slot in slots
    }

    source_counts = {slot: {} for slot in slots}

    for source, slot_map in all_candidates.items():
        for slot, numbers in slot_map.items():
            source_counts.setdefault(slot, {})[source] = len(numbers)
            seen = set()
            deduped = []
            for num in numbers:
                if num in seen:
                    continue
                seen.add(num)
                deduped.append(num)

            for rank, num in enumerate(deduped):
                agg = aggregated[slot][num]
                agg["votes"] += 1
                agg["sources"].add(source)
                # Base vote weight + a tiny rank bonus to keep script order visible
                rank_bonus = 0.1 / (rank + 1)
                weight = (1.0 + rank_bonus) * _regime_multiplier(slot, num, regime_states)
                agg["score"] += weight

    return aggregated, source_counts


def summarize_results(aggregated, learning_signals=None):
    summary = {}
    for slot in ["FRBD", "GZBD", "GALI", "DSWR"]:
        entries = []
        slot_signal = learning_signals.get(slot) if learning_signals else None
        slot_details = slot_signal.get("details", {}) if slot_signal else {}
        for num, info in aggregated[slot].items():
            num_int = int(num)
            signal_tags = slot_details.get(num_int, [])
            entries.append(
                {
                    "number": num,
                    "score": info["score"],
                    "votes": info["votes"],
                    "sources": sorted(info["sources"]),
                    "learning_signals": "; ".join(signal_tags),
                }
            )

        entries.sort(key=lambda x: (-x["score"], -x["votes"], int(x["number"])))
        summary[slot] = entries

    return summary


def _format_andar_bahar(numbers: List[str]):
    """Return the most common tens/ones digits from the shortlist."""

    if not numbers:
        return None, None

    tens_counter = defaultdict(int)
    ones_counter = defaultdict(int)
    for num in numbers:
        n = int(num)
        tens_counter[n // 10] += 1
        ones_counter[n % 10] += 1

    andar = max(tens_counter.items(), key=lambda x: (x[1], x[0]))[0]
    bahar = max(ones_counter.items(), key=lambda x: (x[1], x[0]))[0]
    return andar, bahar


def _split_tiers(entries, top_n=15):
    """Split top-N entries into A/B/C tiers for printing & saving.
    Tiering rule (stable):
      - A: first 5
      - B: next 5
      - C: next 5
    If top_n < 15, tiers shrink accordingly.
    Returns dict: {'A': [...], 'B': [...], 'C': [...]}
    """
    if not entries:
        return {"A": [], "B": [], "C": []}

    top_n = int(top_n or 0)
    if top_n <= 0:
        return {"A": [], "B": [], "C": []}

    top_entries = list(entries)[:top_n]

    a_n = min(5, len(top_entries))
    b_n = min(5, max(0, len(top_entries) - a_n))
    c_n = max(0, len(top_entries) - a_n - b_n)

    tiers = {
        "A": top_entries[:a_n],
        "B": top_entries[a_n:a_n + b_n],
        "C": top_entries[a_n + b_n:a_n + b_n + c_n],
    }
    return tiers

def _print_tiers(shortlisted: List[Dict], *, top_n: int):
    """Print tiered shortlist view with just the numbers for readability."""

    tiers = {
        "A (High)": shortlisted[: min(5, len(shortlisted))],
        "B (Medium)": shortlisted[min(5, len(shortlisted)) : min(10, len(shortlisted))],
        "C (Spec)": shortlisted[min(10, len(shortlisted)) : top_n],
    }

    for name, rows in tiers.items():
        if not rows:
            continue
        pretty = [r["number"] for r in rows]
        print(f"   {name}: " + ", ".join(pretty))


def print_results(
    summary,
    source_counts,
    successes,
    failures,
    *,
    debug: bool,
    shortlist_map: Dict[str, List[Dict[str, object]]],
    overlay_report: Optional[Dict[str, Dict[str, object]]] = None,
    regime_notes=None,
):
    for slot, entries in summary.items():
        shortlist = shortlist_map.get(slot, entries)
        header = (
            f"SLOT: {slot} "
            f"top{len(shortlist)} of {len(entries)} "
            f"scripts_used={len(successes)}/{len(successes) + len(failures)}"
        )
        print(header)

        diagnostics = [f"{src}:{source_counts.get(slot, {}).get(src, 0)}" for src in successes]
        if diagnostics:
            print("   sources -> " + ", ".join(diagnostics))

        if regime_notes:
            note = regime_notes.get(slot)
            if note:
                print(f"   Regimes → {note}")

        overlay_info = (overlay_report or {}).get(slot, {})
        overlay_added = overlay_info.get("added") or []
        overlay_status = overlay_info.get("status")
        if overlay_status or overlay_added:
            added_nums = ",".join(row["number"] for row in overlay_added)
            contrib = f" add:{added_nums}" if added_nums else ""
            print(f"   Overlays → {overlay_status or 'n/a'}{contrib}")

        shortlisted = shortlist
        if not shortlisted:
            print("-")
            print()
            continue

        final_picks = [row["number"] for row in shortlisted[: min(5, len(shortlisted))]]
        print(f"   Final Top {len(final_picks)}: " + ", ".join(final_picks))

        _print_tiers(shortlisted, top_n=len(shortlisted))

        signal_notes = [
            f"{row['number']}[{row['learning_signals']}]"
            for row in shortlisted
            if row.get("learning_signals")
        ]
        if signal_notes:
            print(f"   Learning signals → {'; '.join(signal_notes)}")

        andar, bahar = _format_andar_bahar(final_picks)
        if andar is not None and bahar is not None:
            print(f"   ANDAR/BAHAR → Tens:{andar} Ones:{bahar}")

        print()

        if debug and len(entries) > len(shortlisted):
            shortlisted_ids = {row["number"] for row in shortlisted}
            remaining = [row for row in entries if row["number"] not in shortlisted_ids]
            compact = [
                f"{row['number']}(v={row['votes']} s={row['score']:.2f})"
                for row in remaining
            ]
            if compact:
                print("   [debug extra] " + " ".join(compact))
            print()


def save_summary(summary, out_dir, shortlist_map, regime_notes=None, overlay_report=None):
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for slot, entries in summary.items():
        slot_note = None
        if regime_notes:
            slot_note = regime_notes.get(slot)
        shortlist = shortlist_map.get(slot, entries)
        tiers = _split_tiers(entries, top_n=len(shortlist))  # tiers apply to selected set only
        # Build a quick lookup: number -> tier (A/B/C) for shortlisted set
        tier_map = {}
        for tname, tlist in tiers.items():
            for r in tlist:
                tier_map[r["number"]] = tname

        shortlist_numbers = {row["number"] for row in shortlist}
        overlay_info = (overlay_report or {}).get(slot, {})
        overlay_status = overlay_info.get("status", "")
        overlay_family_map = {
            row["number"]: row.get("overlay_family", "") for row in overlay_info.get("added", [])
        }

        for rank, row in enumerate(entries, start=1):
            num = row["number"]
            rows.append(
                {
                    "slot": slot,
                    "rank": rank,
                    "number": num,
                    "tier": tier_map.get(num, "OTHER"),
                    "in_top": 1 if num in shortlist_numbers else 0,
                    "votes": row["votes"],
                    "score": row["score"],
                    "sources": ",".join(row["sources"]),
                    "learning_signals": row.get("learning_signals", ""),
                    "regime_notes": slot_note or "",
                    "overlay_family": overlay_family_map.get(num, ""),
                    "overlay_status": overlay_status,
                }
            )

    df = pd.DataFrame(rows)
    out_csv = os.path.join(out_dir, "scr9_shortlist.csv")
    out_xlsx = os.path.join(out_dir, "scr9_shortlist.xlsx")
    df.to_csv(out_csv, index=False)
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="shortlist_all")

    return out_csv, out_xlsx
def main():
    parser = argparse.ArgumentParser(description="Run SCR1-8 aggregation (SCR9)")
    parser.add_argument("--debug", action="store_true", help="Print full debug dump")
    parser.add_argument("--top", type=int, default=15, help="Top N results to display")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("predictions/deepseek_scr9"),
        help="Directory to store shortlist exports",
    )
    parser.add_argument(
        "--per-slot-limit",
        type=int,
        default=30,
        help="Cap per-script numbers per slot when reading files",
    )
    args = parser.parse_args()

    # Reset log file
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write("")

    base_df = load_base_dataframe()
    df_long = to_long_format(base_df)

    date_min = df_long["date"].min()
    date_max = df_long["date"].max()
    print(f"Loaded {len(df_long)} rows | {date_min:%Y-%m-%d} -> {date_max:%Y-%m-%d}")

    regime_states = compute_regime_states()
    regime_notes = {slot: format_regime_note(regime_states.get(slot, {})) for slot in ["FRBD", "GZBD", "GALI", "DSWR"]}
    learning_signals = compute_learning_signals(base_df, target_date=(date_max + pd.Timedelta(days=1)).date())

    all_candidates = {}
    successes = []
    failures = []
    script_dir = Path(__file__).parent

    for key, adapter in ADAPTERS.items():
        slot_map = {"FRBD": [], "GZBD": [], "GALI": [], "DSWR": []}
        loaded = False

        latest = latest_glob(SCRIPT_PATTERNS.get(key, []), script_dir)
        if latest:
            try:
                slot_map = read_predictions_any(
                    latest, target_date=date_max, per_slot_limit=args.per_slot_limit
                )
                loaded = True
                log(f"{key} loaded from file {latest}", level="INFO")
            except Exception as exc:
                log(f"{key} file read failed: {exc}", level="ERROR")
                log(traceback.format_exc(), level="ERROR")

        if not loaded:
            try:
                with open(LOG_PATH, "a", encoding="utf-8") as log_file:
                    with redirect_stdout(log_file), redirect_stderr(log_file):
                        slot_map = adapter(df_long)
                loaded = True
                log_debug(f"{key} adapter collected { {k: len(v) for k, v in slot_map.items()} }")
            except Exception as exc:
                log(f"{key} adapter failed: {exc}", level="ERROR")
                log(traceback.format_exc(), level="ERROR")

        if loaded:
            successes.append(key)
        else:
            failures.append(key)

        all_candidates[key] = slot_map

    aggregated, source_counts = aggregate_candidates(all_candidates, regime_states)
    summary = summarize_results(aggregated, learning_signals=learning_signals)
    shortlist_map = {
        slot: _apply_selection_rules(entries, base_limit=args.top)
        for slot, entries in summary.items()
    }
    shortlist_map, overlay_report = _apply_overlays(summary, shortlist_map, regime_states)

    print(
        f"Scripts ok: {', '.join(successes) if successes else '-'} | "
        f"failed: {', '.join(failures) if failures else '-'}"
    )
    print()
    print_results(
        summary,
        source_counts,
        successes,
        failures,
        debug=args.debug,
        shortlist_map=shortlist_map,
        overlay_report=overlay_report,
        regime_notes=regime_notes,
    )
    save_summary(
        summary,
        args.output_dir,
        shortlist_map,
        regime_notes=regime_notes,
        overlay_report=overlay_report,
    )

    # Acceptance: ensure deduped union is not artificially tiny
    for slot in ["FRBD", "GZBD", "GALI", "DSWR"]:
        max_single = max((len(all_candidates[src].get(slot, [])) for src in all_candidates), default=0)
        unique_count = len(aggregated[slot])
        assert unique_count >= max(max_single, 10), (
            f"Slot {slot} union too small: unique={unique_count}, max_single={max_single}"
        )


if __name__ == "__main__":
    main()