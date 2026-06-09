from __future__ import annotations

from common.object.serializer import Serializable
from common.object.validation import Validatable


class CommonObject(Serializable, Validatable):
    """Base class for domain objects providing structured validation and serialization.

    Composes ``Serializable`` (JSON round-trip with dtype-aware numpy and
    polymorphic ``__type__`` registry) and ``Validatable`` (structured
    ``_validate_impl`` → ``ValidationResult`` contract).
    """
