"""
line1d_construction expects - Regression Test
"""

ratios = '''
| benchmark (str)    | impl (str)               |   ratio (float64) |
|:-------------------|:-------------------------|------------------:|
| construct 50 nodes | Line1d(Linear) fresh     |    5.549521219917 |
| construct 50 nodes | Line1d(Linear).with_y    |    4.406864965771 |
| construct 50 nodes | Line1d(Cubic) fresh      |   31.832439909685 |
| construct 50 nodes | Line1d(Cubic).with_y     |   29.471578973573 |
| construct 50 nodes | scipy CubicSpline alone  |   28.332721448686 |
| construct 50 nodes | Line1d(Quadratic) fresh  |   11.199774113676 |
| construct 50 nodes | Line1d(Quadratic).with_y |    9.898142170153 |
'''
