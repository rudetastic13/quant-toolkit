from common.containers.enums import SupportedIntEnum

class MarginTreatment(SupportedIntEnum):
    """Enum for different types of margin treatment."""
    Unused = -1
    Inclusive = 0
    Exclusive = 1
    Flat = 2
