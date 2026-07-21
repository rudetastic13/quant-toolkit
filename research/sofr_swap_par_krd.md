## Par-rate key-rate duration — receive-fixed SOFR swaps ($100mm, struck at par)

Market date 2026-06-03.  Values are PV change in USD per **+1bp** move in each par calibration quote, via the calibration Jacobian (JAX autodiff).

| par quote    |   5Y recv-fixed @ 3.9385% |   7Y recv-fixed @ 3.9924% |   10Y recv-fixed @ 4.0885% |   5Y recv-fixed @ 3.8000% (off-mkt) |
|:-------------|--------------------------:|--------------------------:|---------------------------:|------------------------------------:|
| 12M          |                       0.0 |                       0.0 |                        0.0 |                                11.4 |
| 24M          |                       0.0 |                       0.0 |                        0.0 |                                23.0 |
| 36M          |                      -0.0 |                       0.0 |                        0.0 |                                35.2 |
| 48M          |                       0.0 |                      -0.0 |                       -0.0 |                                47.9 |
| 60M          |                 -45,167.8 |                       0.1 |                        0.0 |                           -45,106.8 |
| 84M          |                       0.0 |                 -60,879.4 |                       -0.0 |                                 0.0 |
| 120M         |                       0.0 |                       0.0 |                  -82,000.3 |                                 0.0 |
| Total (DV01) |                 -45,167.8 |                 -60,879.3 |                  -82,000.2 |                           -44,989.1 |
