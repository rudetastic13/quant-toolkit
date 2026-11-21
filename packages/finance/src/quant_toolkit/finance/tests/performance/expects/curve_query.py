"""
curve_query expects - Regression Test
"""

ratios = '''
| benchmark (object)        | impl (object)                    |   ratio (float64) |
|:--------------------------|:---------------------------------|------------------:|
| DF query 100k, 40 pillars | ZeroCurve LogDF/Linear           |    1.095527728755 |
| DF query 100k, 40 pillars | YieldCurve (dates in)            |    1.230796433557 |
| DF query 100k, 40 pillars | dates_to_x alone                 |    0.091500346266 |
| DF query 100k, 40 pillars | ZeroCurve LogDF/Cubic            |    1.343397698970 |
| DF query 100k, 40 pillars | ZeroCurve ZeroRate/Linear        |    1.332842338175 |
| DF query 100k, 40 pillars | ZeroCurve zero_rate LogDF/Linear |    1.319401018332 |
'''
