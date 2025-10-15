from enum import StrEnum

class TermType(StrEnum):
    Days = "D"
    Weeks = "W"
    Months = "M"
    Quarters = "Q"
    Years = "Y"
    BusinessDays = "BD"