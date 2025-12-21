# Legacy Daily Report Implementation Summary

## Overview
This implementation adds complete legacy Daily Report sections to the `--generate-future` mode output. When running with the `--legacy-display` flag, the system now displays all analytical sections after predictions are generated.

## Changes Made

### 1. New Function: `_print_legacy_daily_report_for_future()`
**Location:** `run_minimal_console.py` lines 2670-2840

**Purpose:** Generate and print the complete legacy daily report for future predictions

**Sections Included:**
1. **DAILY REPORT Header** - Visual separator with report title
2. **Yesterday's P&L Table** - Detailed profit/loss breakdown by slot
3. **Best Top-K Strategy Yesterday** - Optimal picking strategy analysis
4. **Andar/Bahar Gate for Next Day** - Gating recommendations based on historical ROI
5. **Rank Buckets & K-AUTO (30D)** - Distribution analysis and automatic limits
6. **Tag ROI (30D) & Booster** - Script tag performance and boosting logic
7. **Cross-Slot Hits (60D)** - Sequential pattern analysis across time slots
8. **Hero/Avoid Scripts (30D)** - Script performance recommendations
9. **Performance Rollups** - Summary metrics (day, 7D, month, cumulative)
10. **Notes Section** - System notes and warnings

**Report Saving:** Saves to `reports/daily_report_auto.txt`

### 2. Modified Function: `generate_future_predictions_range()`
**Changes:**
- Added `legacy_display` parameter (default: False)
- Updated docstring with details of all report sections
- Moved `results_df` loading outside try block for reliable access
- Added conditional call to `_print_legacy_daily_report_for_future()` when `legacy_display=True`

### 3. Updated Command-Line Arguments
**Location:** `run_minimal_console.py` lines 2280-2299

**New Arguments:**
- `--legacy-display`: Enable legacy daily report display (default: ON)
- `--no-legacy-display`: Disable legacy daily report display

**Behavior:**
- Default is to show the legacy display
- Users can explicitly disable with `--no-legacy-display`

### 4. Updated All Function Calls
All calls to `generate_future_predictions_range()` now pass the `legacy_display` parameter:
- From `handle_future_generation_mode()` for custom date range
- From `handle_future_generation_mode()` for --until mode  
- From interactive menu options (backfill + next day)

## Usage Examples

### Generate future predictions with legacy display (default)
```bash
py -3.12 run_minimal_console.py --generate-future
py -3.12 run_minimal_console.py --generate-future --legacy-display
py -3.12 run_minimal_console.py --generate-future --from 01-04-25 --to 20-12-25
```

### Generate future predictions without legacy display
```bash
py -3.12 run_minimal_console.py --generate-future --no-legacy-display
py -3.12 run_minimal_console.py --generate-future --from 01-04-25 --to 20-12-25 --no-legacy-display
```

### With other options
```bash
# With metrics rebuild
py -3.12 run_minimal_console.py --generate-future --rebuild-metrics --legacy-display

# With custom AB cutoff
py -3.12 run_minimal_console.py --generate-future --ab-cutoff prev --legacy-display

# Verbose mode
py -3.12 run_minimal_console.py --generate-future --verbose --legacy-display
```

## Expected Output Format

When `--legacy-display` is enabled, the output for each day includes:

```
📅 Day 1/N: Generating predictions for DD-MM-YY
--------------------------------------------------

✅ STRONGEST ANDAR/BAHAR DIGITS
FRBD → 56 (tens:5, ones:6)
GZBD → 99 (tens:9, ones:9)
GALI → 78 (tens:7, ones:8)
DSWR → 38 (tens:3, ones:8)

📊 FINAL BET NUMBERS
FRBD (20): 56 23 45 67 89 12 34 56 78 90 11 22 33 44 55 66 77 88 99 00
GZBD (20): 99 88 77 66 55 44 33 22 11 00 12 23 34 45 56 67 78 89 90 01
GALI (20): 78 56 34 12 90 67 45 23 01 89 77 55 33 11 99 88 66 44 22 00
DSWR (20): 38 49 60 71 82 93 04 15 26 37 48 59 70 81 92 03 14 25 36 47

💰 Total Picks: 80 | Expected Stake: ₹800

==================================================
DAILY REPORT
==================================================

💰 YESTERDAY'S P&L (17-07-25)
Slot    Result  Picks  Stake  Return   P&L     ROI      AB    AB P&L
----    ------  -----  -----  ------   ----    ---      --    ------
FRBD    56         20    200     900   +700   +350%     -    +0  [HIT]
GZBD    99         20    200       0   -200   -100%     -    +0  [MISS]
GALI    78         20    200       0   -200   -100%     B   +70  [MISS]
DSWR    38         20    200       0   -200   -100%     -    +0  [MISS]
----------------------------------------------------------------------
Total:              80    800     900   +100    +12%          +70

BEST TOP-K STRATEGY YESTERDAY
Best: K15 → Stake 600 | Return 900 | P&L +300 | ROI +50%
Winning Slot: FRBD (K15, HIT, rank 15)

ANDAR/BAHAR GATE FOR NEXT DAY (based on data up to 2025-07-17)
Slot  Gate  7D ROI  30D ROI  ALL ROI  Hit7D
----  ----  -------  -------  -------  -----
FRBD  [OFF]    -100%    -55%    -61%    0%
GZBD  [OFF]     -36%    -25%    -32%   14%
GALI  [ON]      +93%    +20%    -12%   29%
DSWR  [OFF]     +29%    -25%     -2%   29%

🔢 RANK BUCKETS & K-AUTO (30D)
Slot    Top10  11-15  16-20  21-25  26-30  31-40 K-AUTO
FRBD        6      3      0      2      0      0      20
GZBD        2      5      1      0      0      0      20
GALI        3      0      5      0      3      0      20
DSWR        1      2      5      0      0      1      20

TAG ROI (30D) & BOOSTER
Slot | S40 ROI | 164950 ROI | BOTH ROI | Booster
--------------------------------------------------
FRBD |  +1058% |      +1585% |    +1054% | [ON]
GZBD |  +1103% |       +614% |     +620% | [ON]
GALI |   +990% |       +420% |     -100% | [OFF]
DSWR |  +1153% |       +162% |     +362% | [OFF]

CROSS-SLOT HITS (60D)
FRBD → FRBD (20), GALI (17)
GZBD → DSWR (20), FRBD (19)
GALI → FRBD (19), GALI (17)
DSWR → FRBD (18), DSWR (17)

HERO/AVOID SCRIPTS (30D)
Slot   Script      Bets  ROI(30D)  Status      Note
----   ------      ----  --------  ------      ----
FRBD   scr7         142    +3069%  HERO        Strong
FRBD   scr9_merged  309    +1356%  HERO        Strong
FRBD   scr8         208     -100%  AVOID       Loss-making
...

📈 PERFORMANCE ROLLUPS
Period      Status  Stake     P&L        ROI
-------     ------  -----     ----       ---
Day         [OK]    880       +110       +12.5%
7D          [GOOD]  6.3K      +1430      +22.7%
Month       [BAD]   16.9K     -1420      -8.4%
Cumulative  [OK]    58.8K     +4210      +7.2%

📝 NOTES
• Metrics: prebuilt older than latest results

✅ Daily report saved to: reports/daily_report_auto.txt
✅ Predictions saved for 17-07-25
```

## Technical Details

### Data Loading
- Loads historical results from `load_results_dataframe()`
- Filters out month-end dates
- Uses last 30 days of data for metrics computation
- Falls back gracefully if historical data is missing

### P&L Computation
- Loads prediction history from `_load_prediction_history()`
- Filters to dates with actual results
- Computes comprehensive P&L report using `compute_pnl_report()`
- Generates actual result mapping for yesterday's performance

### Report Generation
- Uses existing helper functions from `bet_pnl_tracker` module:
  - `format_rank_bucket_windows()` - Rank buckets & K-AUTO
  - `format_topk_profit()` - Best Top-K strategy
  - `format_andar_bahar_gating()` - AB gating logic
  - `format_tag_roi()` - Tag ROI & Booster
  - `format_cross_slot_hits()` - Cross-slot patterns
  - `format_hero_weakest()` - Hero/Avoid scripts
- Formats rollups using `_format_rollups()`
- Saves complete report using `_render_daily_report()` and `_write_text()`

### Error Handling
- Wrapped in try-except to prevent crashes
- Provides informative warnings when data is missing
- Falls back gracefully without breaking prediction generation
- Verbose mode shows additional diagnostic information

## Testing

Since the repository lacks test infrastructure, manual testing is recommended:

1. **Test with historical data:**
   ```bash
   py -3.12 run_minimal_console.py --generate-future --from 01-07-25 --to 05-07-25 --legacy-display
   ```

2. **Test without legacy display:**
   ```bash
   py -3.12 run_minimal_console.py --generate-future --from 01-07-25 --to 05-07-25 --no-legacy-display
   ```

3. **Verify report file creation:**
   - Check that `reports/daily_report_auto.txt` is created
   - Verify all sections are present in the saved file

4. **Test with verbose mode:**
   ```bash
   py -3.12 run_minimal_console.py --generate-future --verbose --legacy-display
   ```

## Files Modified
- `run_minimal_console.py` - Main implementation file (added ~200 lines)

## Backward Compatibility
- Default behavior includes legacy display (maintains user expectations)
- Users can opt-out with `--no-legacy-display`
- No breaking changes to existing functionality
- All existing command-line options continue to work

## Next Steps
1. Test with real data when available
2. Gather user feedback on report format
3. Consider adding configuration options for report sections
4. Add unit tests when test infrastructure is established
