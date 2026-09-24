"""
line1d_query expects - Regression Test
"""

ratios = """
| benchmark (object)               | impl (object)            |   ratio (float64) |
|:---------------------------------|:-------------------------|------------------:|
| query 100k: Linear               | Line1d(Linear)           |    1.194431573675 |
| query 100k: Flat (previous-hold) | Line1d(Flat)             |    1.163202956148 |
| query 100k: Cubic                | Line1d(Cubic)            |    1.215658747593 |
| query 100k: Cubic                | Line1d(Cubic).derivative |    1.413806579159 |
"""
