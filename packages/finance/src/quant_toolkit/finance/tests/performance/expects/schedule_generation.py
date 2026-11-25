"""
schedule_generation expects - Regression Test
"""

ratios = '''
| benchmark (str)                                  | impl (str)      |   ratio (float64) |
|:-------------------------------------------------|:----------------|------------------:|
| Annually: 2025-01-01 -> 2029-01-01, 5 dates      | while-loop (py) |    5.821110975370 |
| Annually: 2025-01-01 -> 2029-01-01, 5 dates      | vectorized (np) |   10.563192098340 |
| Annually: 2025-01-01 -> 2029-01-01, 5 dates      | numba @njit     |    1.893329593365 |
| SemiAnnually: 2025-01-01 -> 2029-07-01, 10 dates | while-loop (py) |    9.242563603819 |
| SemiAnnually: 2025-01-01 -> 2029-07-01, 10 dates | vectorized (np) |    9.947189764400 |
| SemiAnnually: 2025-01-01 -> 2029-07-01, 10 dates | numba @njit     |    1.674453786083 |
| Quarterly: 2025-01-01 -> 2029-10-01, 20 dates    | while-loop (py) |   15.688885250758 |
| Quarterly: 2025-01-01 -> 2029-10-01, 20 dates    | vectorized (np) |    9.516304010850 |
| Quarterly: 2025-01-01 -> 2029-10-01, 20 dates    | numba @njit     |    1.638815950197 |
| Monthly: 2025-01-01 -> 2029-12-01, 60 dates      | while-loop (py) |   35.144066366351 |
| Monthly: 2025-01-01 -> 2029-12-01, 60 dates      | vectorized (np) |    7.669151780787 |
| Monthly: 2025-01-01 -> 2029-12-01, 60 dates      | numba @njit     |    1.482822934358 |
| Weekly: 2025-01-01 -> 2029-12-26, 261 dates      | while-loop (py) |   31.081652619726 |
| Weekly: 2025-01-01 -> 2029-12-26, 261 dates      | vectorized (np) |    6.801751626303 |
| Weekly: 2025-01-01 -> 2029-12-26, 261 dates      | numba @njit     |    2.314313544033 |
| Daily: 2025-01-01 -> 2029-12-31, 1826 dates      | while-loop (py) |  127.240544110987 |
| Daily: 2025-01-01 -> 2029-12-31, 1826 dates      | vectorized (np) |    5.449019348688 |
| Daily: 2025-01-01 -> 2029-12-31, 1826 dates      | numba @njit     |    1.655098002492 |
'''
