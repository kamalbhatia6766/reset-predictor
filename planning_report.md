# Planning Report

## Progress Update
- Reverted the working tree to the PR #91 baseline so the interactive daily console display and legacy reporting flows are restored.
- Removed post-PR91 changes (consensus voting, logger-to-print changes, and follow-on tweaks) by rolling back to the earlier snapshot.
- Confirmed PR #91 scope remains the current target for ongoing backtest and console behavior.
- Began locking the generate-future console output template in `run_minimal_console.py` while routing extra status lines to the verbose tail.
- Added a guard to exit cleanly when the results file is missing or lacks DATE columns.

## Current Issues / Risks
- Full backtest execution still depends on local availability of `pandas`/`openpyxl` for replaying historical data in this environment.
- Historic generation relies on scr1–scr9 honoring the shared cutoff environment variable; needs runtime validation in the target environment.
- AB gate computation depends on prediction history availability in `scr9_shortlist_history.csv`; missing history may mute AB staking.
- Windows scheduled task installation requires running `tools/install_scheduler.cmd` once with appropriate permissions.
- Rollback to PR #91 means any post-rollback fixes must be re-applied deliberately if they are still required.
- Generate-future relies on prebuilt metrics availability; if prebuilt files are missing, verbose warnings will be the only clue unless metrics are rebuilt.

## Roadmap (Next Steps)
- Re-run `py -3.12 run_minimal_console.py --backtest --last 3 --verbose` to confirm PR #91 baseline behavior is intact.
- Validate `py -3.12 run_minimal_console.py --generate-future --until 21-07-25 --verbose` on the reverted build for range accuracy.
- Re-check any previously expected self-tests or ROI banners to ensure they match the PR #91 behavior.
- Execute the new generate-future commands (`--rebuild-metrics`, `--from/--to`, `--until`) to confirm the locked output headers stay byte-for-byte.
