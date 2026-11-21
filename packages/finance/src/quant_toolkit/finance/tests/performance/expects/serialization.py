"""
serialization expects - Regression Test
"""

ratios = '''
| benchmark (object)                      | impl (object)   |   ratio (float64) |
|:----------------------------------------|:----------------|------------------:|
| document: swap                          | loads           |    8.906058865780 |
| document: swap                          | dumps           |    4.478152443751 |
| document: instrument, 118 coupon events | loads           |    8.855803229281 |
| document: instrument, 118 coupon events | dumps           |    5.191548194044 |
'''
