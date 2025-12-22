# Planning Report

## Progress Update
- Centralized daily P&L calculations into `bet_pnl_tracker.compute_daily_pnl_summary` and wired generate-future/backtest to reuse the same engine.
- Adjusted generate-future performance rollups to sum the same daily P&L series used in the “YESTERDAY’S P&L” block, avoiding mismatched rollup sources.
- Kept console output formatting intact while switching daily/backtest math to the shared helper.

## Current Issues / Risks
- Full backtest execution still depends on local availability of `pandas`/`openpyxl` for replaying historical data in this environment.
- Historic generation relies on scr1–scr9 honoring the shared cutoff environment variable; needs runtime validation in the target environment.
- AB gate computation depends on prediction history availability in `scr9_shortlist_history.csv`; missing history may mute AB staking.
- Rollup reconstruction now depends on saved per-day shortlists; missing daily folders will skip those days in rollups.

## Roadmap (Next Steps)
- Run the specified backtest/generate-future commands to confirm daily P&L and rollups align for the same dates.
- Spot-check rollups for dates with missing saved shortlists to ensure the skips are acceptable and messaging remains clear.
