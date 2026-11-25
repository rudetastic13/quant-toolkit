"""
curve_rebind expects - Regression Test
"""

ratios = '''
| benchmark (str)                      | impl (str)                       |   ratio (float64) |
|:-------------------------------------|:---------------------------------|------------------:|
| rebind 40 pillars (calibration step) | ZeroCurve(LogDF/Linear) fresh    |   17.763864975657 |
| rebind 40 pillars (calibration step) | ZeroCurve(LogDF/Linear).with_dfs |   15.071902872755 |
| rebind 40 pillars (calibration step) | ZeroCurve(LogDF/Cubic) fresh     |   64.392362068606 |
| rebind 40 pillars (calibration step) | ZeroCurve(LogDF/Cubic).with_dfs  |   61.281888343294 |
'''
