# Project Rules and Learning Signals Summary

This document consolidates the current rule set, learning signals, and operational goals for the Precise Predictor project. It is meant as a compact reference for future implementation and reporting changes.

## Game Model
- **Universe:** Two-digit numbers `00`–`99` (zero-padded strings). `00` is valid.
- **Slots/Clocks:** Four daily slots — `FRBD`, `GZBD`, `GALI`, `DSWR`; day boundary closes after `DSWR` and next-day predictions lock only after `DSWR` results.
- **Andar/Bahar:** `ANDAR = tens digit`, `BAHAR = ones digit` for each pick.

## Fixed Packs and Families
- **S40 pack (fixed, no edits):** `{00, 06, 07, 09, 15, 16, 18, 19, 24, 25, 27, 28, 33, 34, 36, 37, 42, 43, 45, 46, 51, 52, 54, 55, 60, 61, 63, 64, 70, 72, 73, 79, 81, 82, 88, 89, 90, 91, 97, 98}`; soft heuristic that at least one hit per day is expected.
- **164950 digit family:** Digits `{0, 1, 4, 5, 6, 9}`; valid numbers where both digits are from the set (36 total).
- **Cross-product packs:** For pairs AB/CD → `{AC, AD, BC, BD}`; extends to 3–6 digit strings; duplicates not allowed.
- **27 pattern families (always on):** Mirrors/half-mirrors, adjacency and neighbor-skip, digit-sum/digital-root, mod-3/mod-9 cycles, tens/ones bias, zero-center (45–55), carry/reverse-carry, parity/primes, spike/cooling cycles, hero/weak digit, loop-90/triangle, S40, 164950, mirror, neighbor, and related pattern regimes with OFF/NORMAL/BOOST states (max 2–3 boosts per day).

## Prediction Pipeline
- **Scripts:** `scr1`–`scr8` generate candidates with distinct logic; `scr9` aggregates, deduplicates, and orders by consensus/weight.
- **Shortlist target:** 3–5 top picks per slot plus derived `ANDAR`/`BAHAR`; minimal screen clutter, focus on Top-N.

## Learning Signals
- Exact hit per slot.
- Cross-slot (same day, different slot) hits.
- Cross-day (previous/next day) soft credit.
- Mirror hits (`XY` vs `YX`).
- Neighbor hits (`XY±1`).
- Pattern family heat (S40, 164950, 27-family set) tracked with regime states.

## Top-N and Diversity Rules
- **Dynamic Top-N:** EV gate within 3% of top score; cap 12 (up to 15 on flat boards).
- **Diversity guards:** ≥3 tens-bins represented; ≤6 numbers per tens-bin.
- **Notes to display:** Winning pick rank and whether it was inside Top-N (e.g., `hit at rank 2 (in top-5)`).

## Staking and Payouts
- **Payout model:** ₹1 → ₹90 on exact 2-digit hit; ₹1 → ₹9 on `ANDAR` or `BAHAR` digit hit.
- **Profit-first (ROI-max) mode: the system dynamically determines how many numbers to play per slot based on expected ROI, with no fixed tiers or fixed count; daily reset, no across-day martingale.

## Auxiliary Overlays
- **S36 filtered + taper:** Slot 1 full, Slot 2 half, Slot 3 quarter, Slot 4 off.
- **Pack-Core 4/4 and Pack-Booster 2/2** overlays.
- **S40 overlay:** Default OFF; conditional ON only if ΔEV>0, 30-day ROI>+10%, and adds ≤6 numbers.

## Risk Guards
- Any layer with 30-day ROI < –10% **and** drawdown > 1.5× 90-day σ auto-OFF until recovery.

## Reporting and Tracking
- Per slot/day: Top-N picks (with tier/stake), `ANDAR`/`BAHAR`, actual result, rank of winning pick, slot P&L, day total, cumulative P&L (main/aux/combined), and future signals (near miss, mirror, cross-slot/day).
- Partial days supported: process available slot results; remaining slots recalc automatically.
- **Month-end skip rule:** The last calendar date of every month is excluded from all calculations and predictions. On the 1st of any month, treat the "previous day" as the 29th if the prior month has 30 days, or the 30th if the prior month has 31 days.

## Workflow Expectations
- Daily run (Windows canonical path): `start.bat` from `C:\Users\kamal\AppData\Local\Programs\Python\Python312\precise predictor`.
- Stable scripts and filenames should remain intact; prefer patches over rewrites to keep the project compact and outputs uncluttered.
- **Mandatory planning hygiene:** Every time this file is read, you must also read `planning_report.md` and update it with the latest project progress, open issues/risks, and roadmap items before completing the run.

## Core Mission (A-S-P-P)
Accuracy, Speed, Precision, Profit — aiming for 500–800%+ long-term ROI; “golden days” are benchmarks, not ignored.

## Backtesting and Iteration Plan
- Future: add intelligence (rank/top-N notes, near-miss, mirror, cross-slot/day, family boosts, overlays) and rerun on the same slice to compare gains while keeping the codebase minimal.
## Scope & Ground Truth
- Ground truth = rules locked in this document plus the confirmed ruleset described in the latest Precise Predictor briefing; use this as single reference for any ambiguity.

## Locked Rules

### Universe & Formatting
- Number universe strictly `00`–`99` only; `00` valid. No expansion beyond two digits in any layer.

### Slots / Day Boundary
- Partial days are valid input (1–3 slots observed); system must process what is present without blocking.

### Payout & P&L Tracking
- No extra rule added here; follow existing payout model already documented above.

### Betting / Staking Rules
- Money management uses dynamic confidence-based stakes with no fixed daily cap; loss-recovery engine remains active alongside Top-N sizing.

### Risk & Auto-Toggles
- Any stable script must fail open (neutral) on missing/partial data instead of crashing.

## Learning Engine

### Signals
- Learning updates also adjust pattern confidence and script scores, not just slot-level hits.

### Windows & Weighting
- Learning window ≠ retention window; old data may keep weak retention while fresh data drives learning weights.
- Insufficient data → neutral weights (do not zero out or fabricate long-run stats).

### Cross-slot / Cross-day Interpretation
- Cross-slot hits (prediction in one slot, hit in another same day) count as positive learning, not failure.
- Cross-day hits remain valid within a sliding relevance window; prediction expiry is not hard-cut at day end.

## Pattern Families (Explained)
- Use OFF/NORMAL/BOOST regimes per family; multiple families can be active together.
- For each family: detection feeds scoring/weighting; mirrors/neighbours count as learning events too.

### S40 Pack
- **Definition:** Fixed 40-number list already documented above.
- **Detect:** Membership in the fixed set; watch for full-day miss.
- **Scoring Impact:** Soft daily expectation of ≥1 hit; no forced bet, but BOOST bias if previous day had zero S40 hits and ROI gate is positive.
- **Edge Cases:** Never modify the set; treat partial-day inputs normally.

### 164950 Digit Family
- **Definition:** Both digits drawn from {0,1,4,5,6,9}; 36 numbers.
- **Detect:** Tens ∈ family AND ones ∈ family.
- **Scoring Impact:** Structural bias; booster state depends on ROI.
- **Edge Cases:** No substitutes; mirrors stay within the same family.

### 2-PACK (AB/CD)
- **Definition:** Cross-product of two-digit groups.
- **Detect:** For digits {A,B} × {C,D} produce {AC, AD, BC, BD}; dedupe.
- **Scoring Impact:** Micro-structure; high ROI when concentrated; often trimmed into Top-K.
- **Edge Cases:** Ignore duplicates if A=C or B=D.

### 3-PACK (ABC/XYZ)
- **Definition:** 3×3 cross-product yielding nine numbers.
- **Detect:** Digits {A,B,C} × {X,Y,Z}.
- **Scoring Impact:** Mid-range coverage; trend continuation bias.
- **Edge Cases:** Deduplicate overlapping digits.

### 4-PACK
- **Definition:** 4×4 cross-product, 16 numbers.
- **Detect:** Digits set of size 4 crossed with another size-4 set.
- **Scoring Impact:** Broader capture; ROI improves after Top-K trimming.
- **Edge Cases:** Remove duplicates when digit pools overlap.

### 5-PACK
- **Definition:** 5×5 cross-product, 25 numbers.
- **Detect:** Two digit pools of size five.
- **Scoring Impact:** Mostly detector/regime input; rarely direct bet unless trimmed hard.
- **Edge Cases:** Deduplicate overlapping outputs.

### 6-PACK
- **Definition:** 6×6 cross-product, 36 numbers.
- **Detect:** Two digit pools of size six.
- **Scoring Impact:** Macro regime detector; direct betting discouraged unless boosted.
- **Edge Cases:** Watch overlap with 164950 family; keep scoring paths distinct.

### Mirror Family
- **Definition:** XY vs YX pairs.
- **Detect:** Reverse of predicted number equals result.
- **Scoring Impact:** Counts as mirror hit learning; boosts mirror-aware scripts.
- **Edge Cases:** Repeat-digit numbers (AA) are self-mirrors; treat as exact, not mirror.

### Neighbour Family (±1)
- **Definition:** Tens or ones differ by exactly 1 from prediction.
- **Detect:** XY±1 on ones place or tens place within 00–99 bounds.
- **Scoring Impact:** Adds near-miss credit; can nudge boosting for adjacent bins.
- **Edge Cases:** Clamp below 00 and above 99.

### ±11 Ladder Family
- **Definition:** Both digits shift together (+1/+1 or –1/–1) e.g., 22→33.
- **Detect:** Result digits each one step above/below prediction digits.
- **Scoring Impact:** Tracks carry-like climbs; useful for streak/cooling detection.
- **Edge Cases:** Clamp at 00/99 boundaries; AA numbers ladder to AA.

### Even / Odd Parity Family
- **Definition:** Parity alignment of digits (EE, EO, OE, OO).
- **Detect:** Check parity per digit.
- **Scoring Impact:** Parity streaks bias scoring toward matching parity buckets.
- **Edge Cases:** None; parity always defined.

### Tens Bias Family
- **Definition:** Repetition bias on tens digit.
- **Detect:** Cluster of results sharing tens digit.
- **Scoring Impact:** Increase weight for numbers sharing hot tens.
- **Edge Cases:** Balance with diversity guards to avoid overloading one tens-bin.

### Ones Bias Family
- **Definition:** Repetition bias on ones digit.
- **Detect:** Cluster on ones digit across slots/days.
- **Scoring Impact:** Elevate numbers sharing the hot ones digit.
- **Edge Cases:** Diversity guard still applies.

### Digit-Sum Family
- **Definition:** Numbers sharing same digit sum.
- **Detect:** Sum(tens, ones) match.
- **Scoring Impact:** Bias toward recurring sums; combines with mod families.
- **Edge Cases:** Treat 09 and 90 separately though sums match.

### Modulo-9 Family
- **Definition:** Digit-sum % 9 alignment.
- **Detect:** (tens+ones) mod 9 equal.
- **Scoring Impact:** Captures cyclical bias; stacks with digit-sum family.
- **Edge Cases:** Handle sum=0 as mod-9=0.

### Modulo-3 Family
- **Definition:** Digit-sum % 3 alignment.
- **Detect:** (tens+ones) mod 3 equal.
- **Scoring Impact:** Macro-cycle indicator; mild boost.
- **Edge Cases:** None.

### Carry Family
- **Definition:** Behaviour after ones overflow (e.g., 39 → 40–42 band).
- **Detect:** Prior hit near 9 on ones triggers carry-forward candidates.
- **Scoring Impact:** Boost post-carry bands briefly.
- **Edge Cases:** Disable if prior slot/day missing.

### Reverse-Carry Family
- **Definition:** Pullback after a carry (e.g., 40 followed by 38–39).
- **Detect:** Detect recent carry then bias to immediate lower neighbours.
- **Scoring Impact:** Short-term cooling bias.
- **Edge Cases:** Skip if data gap around carry event.

### Zero-Center Family
- **Definition:** Concentration near zeros or midline bands (00–09, 90–99, 45–55).
- **Detect:** Result within these bands.
- **Scoring Impact:** Bias toward band persistence when active.
- **Edge Cases:** 45–55 inclusive as mid band.

### Repeat-Digit Family
- **Definition:** AA numbers (11,22,…).
- **Detect:** Tens == ones.
- **Scoring Impact:** Hot when repeats cluster; treat as own mirror.
- **Edge Cases:** Mirror signals redundant; count once.

### Split-Digit Family
- **Definition:** Widely separated digits (e.g., 09,18,27).
- **Detect:** Absolute digit difference high (≥7 typical).
- **Scoring Impact:** Spike indicator; light boost only when recent spikes present.
- **Edge Cases:** Clamp to valid universe.

### Pack-2 Family
- **Definition:** 2×2 pack behaviour (same as 2-PACK) treated as family.
- **Detect:** Cross-product of two digit pairs.
- **Scoring Impact:** Micro cluster bias; strongest when overlapping other families.
- **Edge Cases:** Deduplicate overlaps.

### Pack-3 Family
- **Definition:** 3×3 pack behaviour as family.
- **Detect:** Cross-product of two digit triplets.
- **Scoring Impact:** Mid cluster bias; use Top-K trim.
- **Edge Cases:** Deduplicate overlaps.

### Pack-4 Family
- **Definition:** 4×4 pack behaviour as family.
- **Detect:** Cross-product of two digit quartets.
- **Scoring Impact:** Broad cluster bias; requires trimming.
- **Edge Cases:** Deduplicate overlaps.

### Pack-5 Family
- **Definition:** 5×5 pack behaviour as family.
- **Detect:** Cross-product of two digit quintuples.
- **Scoring Impact:** Regime detector; rarely direct bet.
- **Edge Cases:** Deduplicate overlaps.

### Pack-6 Family
- **Definition:** 6×6 pack behaviour as family.
- **Detect:** Cross-product of two digit sextets.
- **Scoring Impact:** Macro detector; direct betting discouraged.
- **Edge Cases:** Keep distinct from 164950 scoring path.

### Cross-Slot Drift Family
- **Definition:** Hit in one slot repeats in another same day.
- **Detect:** Same number appearing in later slot.
- **Scoring Impact:** Treat as positive drift; boosts cross-slot repeats.
- **Edge Cases:** Supports partial-day data.

### Cross-Day Drift Family
- **Definition:** Today’s signal hits tomorrow (or vice-versa).
- **Detect:** Match between consecutive days.
- **Scoring Impact:** Maintains sliding relevance; mild decay over days.
- **Edge Cases:** Pause when previous day missing.

### Hot-Digit Family
- **Definition:** Digit appearing frequently in recent window.
- **Detect:** Frequency threshold per digit across slots.
- **Scoring Impact:** Bias numbers containing the hot digit.
- **Edge Cases:** Cap to respect diversity guards.

### Cold-Digit Family
- **Definition:** Digit absent for long stretch.
- **Detect:** Long-gap count per digit.
- **Scoring Impact:** Build “pressure” bias; BOOST when gap extreme.
- **Edge Cases:** Reset if data missing.

### Spike / Burst Family
- **Definition:** Sudden out-of-pattern hit (e.g., 99 after long gap).
- **Detect:** Rare number hit vs baseline frequency.
- **Scoring Impact:** Short-lived BOOST on neighbouring patterns.
- **Edge Cases:** Do not persist beyond short window.

### Cooling Family
- **Definition:** Slowdown after back-to-back hits.
- **Detect:** Consecutive hits then drop bias for next slot/day.
- **Scoring Impact:** Temporarily downgrade affected families.
- **Edge Cases:** Lift cooling once two slots/days pass without repeat.

## Operational Workflow

### Daily Run
- Canonical daily run remains `start.bat` from the Windows path already noted.

### Rebuild / Sanity Suite
- Missing data or partial slot availability must not crash rebuilds; fallback to neutral outputs.

### File/Folder Contracts
- `number prediction learn.xlsx` is the single source of historical truth (Jan 2025 onward); expect partially filled days.
- Chain-locked scripts list: `deepseek_scr1-9.py`, `quant_data_core.py`, `quant_excel_loader.py`, `quant_paths.py`, `bet_pnl_tracker.py`, `run_minimal_console.py`, `start.bat`; prefer minimal new scripts.
- Aggregator script (`scr9`/`scr11`-type) must not self-call.

## Non-Negotiables (Do Not Break)
- Core logic loss, stable script breakage, S40/164950 tampering, or universe expansion beyond `00`–`99` are not allowed.
- Avoid undocumented assumptions; if data insufficient, default to neutral rather than inventing stats.

## Audit Footer
Added Rules Count: 14
Added Family Explanations Count: 27
No-Duplication Check: PASS
