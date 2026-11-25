"""
curve_query expects - Regression Test
"""

ratios = '''
| benchmark (str)           | impl (str)                       |   ratio (float64) |
|:--------------------------|:---------------------------------|------------------:|
| DF query 100k, 40 pillars | ZeroCurve LogDF/Linear           |    1.281438952302 |
| DF query 100k, 40 pillars | YieldCurve (dates in)            |    1.379499648272 |
| DF query 100k, 40 pillars | dates_to_x alone                 |    0.098804799912 |
| DF query 100k, 40 pillars | ZeroCurve LogDF/Cubic            |    1.545936574093 |
| DF query 100k, 40 pillars | ZeroCurve ZeroRate/Linear        |    1.495997669432 |
| DF query 100k, 40 pillars | ZeroCurve zero_rate LogDF/Linear |    2.108665672498 |
'''
