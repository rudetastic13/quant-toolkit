from common.containers.enums import SupportedIntEnum


class LegType(SupportedIntEnum):
    """Pay or receive designation for swap legs."""
    Pay = 1
    Receive = 2
