"""
line1d_query expects - Regression Test
"""

ratios = '''
| benchmark (str)                  | impl (str)               |   ratio (float64) |
|:---------------------------------|:-------------------------|------------------:|
| query 100k: Linear               | Line1d(Linear)           |    1.532358568814 |
| query 100k: Flat (previous-hold) | Line1d(Flat)             |    1.257259346019 |
| query 100k: Cubic                | Line1d(Cubic)            |    1.046055584499 |
| query 100k: Cubic                | Line1d(Cubic).derivative |    1.101145762166 |
'''
