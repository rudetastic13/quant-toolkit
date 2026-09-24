"""
line1d_construction expects - Regression Test
"""

ratios = """
| benchmark (object)   | impl (object)            |   ratio (float64) |
|:---------------------|:-------------------------|------------------:|
| construct 50 nodes   | Line1d(Linear) fresh     |    5.253404432377 |
| construct 50 nodes   | Line1d(Linear).with_y    |    4.001023918271 |
| construct 50 nodes   | Line1d(Cubic) fresh      |   23.620112899453 |
| construct 50 nodes   | Line1d(Cubic).with_y     |   21.933403128511 |
| construct 50 nodes   | scipy CubicSpline alone  |   20.157046820293 |
| construct 50 nodes   | Line1d(Quadratic) fresh  |   11.400862085495 |
| construct 50 nodes   | Line1d(Quadratic).with_y |   10.044337195078 |
"""
