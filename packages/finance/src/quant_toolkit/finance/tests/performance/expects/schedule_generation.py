"""
schedule_generation expects - Regression Test
"""

ratios = '''
| benchmark (object)                               | impl (object)   |   ratio (float64) |
|:-------------------------------------------------|:----------------|------------------:|
| Annually: 2025-01-01 -> 2029-01-01, 5 dates      | while-loop (py) |    5.269161200304 |
| Annually: 2025-01-01 -> 2029-01-01, 5 dates      | vectorized (np) |   11.334799567924 |
| Annually: 2025-01-01 -> 2029-01-01, 5 dates      | numba @njit     |    1.891029120889 |
| Annually: 2025-01-01 -> 2029-01-01, 5 dates      | C++ pybind11    |    1.102713077589 |
| SemiAnnually: 2025-01-01 -> 2029-07-01, 10 dates | while-loop (py) |    8.401463239252 |
| SemiAnnually: 2025-01-01 -> 2029-07-01, 10 dates | vectorized (np) |   10.833457078244 |
| SemiAnnually: 2025-01-01 -> 2029-07-01, 10 dates | numba @njit     |    1.830319134225 |
| SemiAnnually: 2025-01-01 -> 2029-07-01, 10 dates | C++ pybind11    |    1.076809423618 |
| Quarterly: 2025-01-01 -> 2029-10-01, 20 dates    | while-loop (py) |   13.435913660029 |
| Quarterly: 2025-01-01 -> 2029-10-01, 20 dates    | vectorized (np) |    9.813900394482 |
| Quarterly: 2025-01-01 -> 2029-10-01, 20 dates    | numba @njit     |    1.765054405209 |
| Quarterly: 2025-01-01 -> 2029-10-01, 20 dates    | C++ pybind11    |    1.035974828198 |
| Monthly: 2025-01-01 -> 2029-12-01, 60 dates      | while-loop (py) |   28.455753956332 |
| Monthly: 2025-01-01 -> 2029-12-01, 60 dates      | vectorized (np) |    7.536273651442 |
| Monthly: 2025-01-01 -> 2029-12-01, 60 dates      | numba @njit     |    1.489263325351 |
| Monthly: 2025-01-01 -> 2029-12-01, 60 dates      | C++ pybind11    |    0.920906973617 |
| Weekly: 2025-01-01 -> 2029-12-26, 261 dates      | while-loop (py) |   29.068574537695 |
| Weekly: 2025-01-01 -> 2029-12-26, 261 dates      | vectorized (np) |    7.070570029466 |
| Weekly: 2025-01-01 -> 2029-12-26, 261 dates      | numba @njit     |    2.500823211018 |
| Weekly: 2025-01-01 -> 2029-12-26, 261 dates      | C++ pybind11    |    1.556400200680 |
| Daily: 2025-01-01 -> 2029-12-31, 1826 dates      | while-loop (py) |  109.748598123522 |
| Daily: 2025-01-01 -> 2029-12-31, 1826 dates      | vectorized (np) |    4.701462963833 |
| Daily: 2025-01-01 -> 2029-12-31, 1826 dates      | numba @njit     |    1.912923108250 |
| Daily: 2025-01-01 -> 2029-12-31, 1826 dates      | C++ pybind11    |    1.246142881757 |
'''
