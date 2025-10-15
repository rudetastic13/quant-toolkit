from enum import StrEnum

class BDC(StrEnum):
    Following = "F"
    ModifiedFollowing = "MF"
    Preceding = "P"
    ModifiedPreceding = "MP"
    NoAdjustment = "N"
