# `common.object`: `Serializable`, `CommonObject`, versioned documents

`Serializable` turns a dataclass into a JSON document and back. `CommonObject` is `Serializable`
plus `Validatable` (structured `_validate_impl` → `ValidationResult`), and is the base for domain
objects that are both saved and checked on construction (`CommonInstrument`).

```python
from dataclasses import dataclass
from common.object import CommonObject, serializable

@serializable("Deposit", version=1)
@dataclass(kw_only=True)
class Deposit(CommonObject):
    notional: float
    rate: float

doc = Deposit(notional=1e6, rate=0.05).serialize()
# {"__type__": "Deposit", "__version__": 1, "notional": 1000000.0, "rate": 0.05}
```

Loading never needs the class. The document names it, and the module-level functions mirror
`json`:

```python
from common.object import dump, dumps, load, loads

text = dumps(deposit)               # JSON text
dump(deposit, "deposit.json")       # to a path or an open file
deposit = loads(text)               # whatever the document describes, by its __type__
deposit = load("deposit.json")
```

`Deposit.deserialize(doc)` and `Deposit.from_json(text)` still exist for the case where you
hold a bare dict with no `__type__` and know what it is.

## When to use it

Use it for **contract data and configuration**: things a person authors, a file holds, and a
later process must read back identically. Instruments and their legs, coupon and notional
schedules, conventions when they become configuration, a saved scenario's description.

Do not use it for **derived or runtime state**: compiled programs, caches, pricing results,
curve objects with fitted interpolators. Those are rebuilt from the contract data. A rule of
thumb: if two fields can disagree after a load, one of them should not be in the document.
`CommonInstrument.serialize` drops a coupon schedule that was derived from flat fields for this
reason, and rebuilds it on load.

Three things must be true of a class you decorate:

- It is a dataclass. Only `init=True` fields are written; `field(init=False)` state is skipped.
- Every field is a primitive, an `Enum`, a numpy array or scalar, a registered value type
  (`Date`, `Term` via `register_type`), another `Serializable`, or a `list` / `dict` / `set` /
  `tuple` of those. Anything else falls back to `str(value)` on write and will not round-trip.
- Type hints are resolvable at runtime: `deserialize` rebuilds each field from its hint, so
  `float | None`, `list[CouponEvent]`, `dict[str, float]` all work; a bare `Any` comes back as
  whatever JSON produced.

## Names

`@serializable("Name")` registers the class under a **stable name**, written as `__type__`.
The name is the contract with every document ever saved:

- Never rename it. Never reuse it for a different class. The class, its module and its package
  are free to move; the loader resolves the name through the registry, not an import path.
- A subclass must register its own name. It does not inherit its parent's, and serialising an
  unregistered `Serializable` is a `TypeError`, so a name is never invented from a module path.
- Short names (`"Swap"`) are fine inside one library. Prefix if two libraries will share
  documents (`"rates.Swap"`). The test suite uses `"test.*"`.
- Loading a document whose `__type__` is not registered is a `KeyError`, never a guess.

## Versions and how to bump one

Each class declares the version it writes and reads natively. A document records it as
`__version__`; a document without one is version 1.

**Bump the version only when an old document can no longer be read by the current fields.**
Adding a field with a default is not a bump: `deserialize` sets only the fields present, so old
documents load with the default. Renaming a field, removing one, changing its meaning or its
units, or splitting one field into two is a bump.

A bump ships with a migration from the previous version, registered next to the class:

```python
@serializable("Deposit", version=2)             # v2: `rate` became `fixed_rate`
@dataclass(kw_only=True)
class Deposit(CommonObject):
    notional: float
    fixed_rate: float

@upgrade("Deposit", from_version=1)
def _deposit_1_to_2(doc: dict) -> dict:
    doc["fixed_rate"] = doc.pop("rate")
    return doc
```

The loader chains migrations: a version 1 document reaching a version 4 class passes through
`1→2`, `2→3`, `3→4`. Each function receives a copy of the previous version's document and
returns the next. Migrations accumulate and are never deleted; a document from any version ever
shipped must load. A document newer than the class is refused with a `ValueError` rather than
read partially.

Nested objects carry their own `__type__` and `__version__`. Bumping `CommonInstrument` does not
bump `Swap`; a saved swap's legs upgrade independently when it loads.

## Fixtures: the discipline that makes the promise real

For every `(name, version)` ever shipped, a frozen document lives under a test `expects/`
directory (`Swap.v1.json`, `Swap.v2.json`, ...). The fixture test loads each file with the
current code and compares it to the object built from the same inputs today. Rules:

- A fixture file is never edited. When you bump a version, add `Name.v<N>.json` and keep the old
  ones.
- A new serialisable type adds a fixture in the same commit.
- If a migration is wrong, the fixture test names the file and the version.

`finance/tests/unit/instruments/test_serialized_fixtures.py` is the model, and also asserts that
every registered finance type has at least one frozen document.

## Value types

A type that is a value, not a document (`Date`, `Term`), registers a pair of converters once:

```python
register_type(Date, lambda d: d.to_str(), Date.from_str)
```

It carries no `__type__`; it is recovered from the field's type hint. Keep these to immutable
scalars whose text form is unambiguous.

## What is deliberately not here

- Graphs of shared objects. The mixin writes a tree. Curves and the market share atoms by
  identity, so they need a memo-based document with references; that is designed and not built.
- Forward compatibility beyond additive change. Old code reading a newer document refuses.
- Schema inference or validation beyond what the type hints and the class's own `__post_init__`
  and `_validate_impl` do on construction.
