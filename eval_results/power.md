# Power simulation (n=142, baseline pass rate 0.873, all split)

Single overall pass-rate test, alpha 0.05, observed drop >= 0.02, 100 simulated candidates per row (each baseline pass flips to fail at random so the expected drop matches).

| true drop | detection power |
|---|---|
| 0.02 | 0.39 |
| 0.03 | 0.66 |
| 0.04 | 0.87 |
| 0.05 | 0.92 |
| 0.06 | 0.95 |
| 0.08 | 0.99 |
| 0.10 | 1.00 |
| 0.12 | 1.00 |
| 0.15 | 1.00 |
| 0.20 | 1.00 |

Smallest drop detected with >= 80% power: **0.04**. Holm correction over ~30 tests makes the per-test bar stricter; drops confined to a small slice are usually only caught by the zero-tolerance critical rule.
