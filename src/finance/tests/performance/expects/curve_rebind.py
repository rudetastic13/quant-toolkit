"""
curve_rebind expects - Regression Test
"""

ratios = '''
| benchmark (object)                   | impl (object)                    |   ratio (float64) |
|:-------------------------------------|:---------------------------------|------------------:|
| rebind 40 pillars (calibration step) | ZeroCurve(LogDF/Linear) fresh    |   16.216690712844 |
| rebind 40 pillars (calibration step) | ZeroCurve(LogDF/Linear).with_dfs |   13.982398838106 |
| rebind 40 pillars (calibration step) | ZeroCurve(LogDF/Cubic) fresh     |   46.214697705963 |
| rebind 40 pillars (calibration step) | ZeroCurve(LogDF/Cubic).with_dfs  |   42.847728515994 |
'''
