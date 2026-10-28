# `research_logs/` – monthly decision log

The strategy is rebalanced monthly. Each month, write down **what the regime looked like and whether you changed anything**, so parameter changes are auditable and can be reversed.

Create `research_logs/YYYY-MM/<name>__rebalance.md` from [`TEMPLATE.md`](TEMPLATE.md).

Rules of thumb:

1. Log the regime indicators (`mr regime`) *before* deciding.
2. Change one parameter at a time and record the old and new value.
3. Record which re-run you did (`cluster --force`, `build-features --force`).
4. Record how you will know the change was wrong, and when you will revert it.
