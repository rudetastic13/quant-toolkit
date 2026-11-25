"""
serialization expects - Regression Test
"""

ratios = '''
| benchmark (str)                         | impl (str)   |   ratio (float64) |
|:----------------------------------------|:-------------|------------------:|
| document: swap                          | loads        |    9.949116964969 |
| document: swap                          | dumps        |    7.367194422987 |
| document: instrument, 118 coupon events | loads        |   10.170617343802 |
| document: instrument, 118 coupon events | dumps        |   11.042908001453 |
'''
