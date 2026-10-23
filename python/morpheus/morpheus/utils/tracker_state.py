# Copyright (c) 2026, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A tracker's memory as plain Python types, so a pipeline can stop and resume without losing its history.

Every per-entity baseline in the telemetry stages -- a port's thirty days of MAC peaks, an optic's week of readings,
a principal's cadence histogram, the novelty sets that decide what is first seen -- lives inside a tracker inside a
stage, in process memory. A process that stops loses all of it, and the one that replaces it spends the next window
relearning: every history-gated rule is silent and every "first seen" fires again. This module is how that memory
leaves the process and comes back.

**The state is the tracker's own attributes, not a summary of them.** A hand-written `to_state` would give every
tracker its own chance to leave a field out, and a field left out is a tracker that resumes almost as it stopped,
which no test over a short corpus would catch. `encode` walks the object instead: its instance attributes, and
theirs, down to plain values. What the tracker holds is what the snapshot holds.

**Order is state, and is kept.** The trackers evict the least recently seen entity, so the order of their entity maps
decides who is dropped next. A mapping is therefore encoded as a list of pairs in its own order, never as a JSON
object whose keys a serializer may sort. Sets have no order a correct tracker may depend on, since string hashing is
randomized per process, and are encoded sorted, so the same set always produces the same bytes.

**Types survive the round trip.** A tuple comes back a tuple, a deque with its bound, a numpy integer as the same
numpy integer, a timestamp with its zone. A tracker that stored `numpy.int64(3)` and resumed holding `3` would emit a
differently typed column, which the golden comparison would catch only after the fact.

**Only Morpheus classes are rebuilt.** A snapshot names the classes it holds, and a loader that imported whatever a
file named would be `pickle` with extra steps. `decode` refuses any class outside the `morpheus` package.

The encoding is JSON-safe: `to_bytes` gives the canonical bytes, and `state_digest` their SHA-256, which is what a
scored row carries to name the state it was scored against.
"""

import base64
import collections
import datetime
import enum
import hashlib
import importlib
import json
import math
import types
import typing

import numpy as np
import pandas as pd

STATE_FORMAT = 1
"""Version of the encoding. A snapshot written under another version is refused rather than misread."""

TAG = "$t"
"""Key marking an encoded value that is not a JSON scalar or list."""

ALLOWED_MODULE_PREFIX = "morpheus."
"""Classes are rebuilt only from modules under this package."""


class StateError(ValueError):
    """A value cannot be encoded, or a snapshot cannot be decoded."""


def _qualified_name(cls: type) -> str:
    return f"{cls.__module__}:{cls.__qualname__}"


def _resolve_attribute(name: str) -> typing.Any:
    (module_name, _, qualname) = name.partition(":")

    if (not module_name.startswith(ALLOWED_MODULE_PREFIX)):
        raise StateError(f"Refusing to rebuild {name!r}: snapshots may only name classes and functions under "
                         f"{ALLOWED_MODULE_PREFIX!r}")

    target: typing.Any = importlib.import_module(module_name)

    for part in qualname.split("."):
        target = getattr(target, part)

    return target


def _resolve_class(name: str) -> type:
    target = _resolve_attribute(name)

    if (not isinstance(target, type)):
        raise StateError(f"{name!r} is not a class")

    return target


def _attributes(obj: typing.Any) -> dict:
    """An object's instance attributes in a fixed order: its `__dict__` as built, then any slots."""
    values = dict(getattr(obj, "__dict__", {}))

    for cls in type(obj).__mro__:
        for slot in getattr(cls, "__slots__", ()):
            if (slot in ("__dict__", "__weakref__") or slot in values):
                continue

            if (hasattr(obj, slot)):
                values[slot] = getattr(obj, slot)

    return values


def _canonical(value: typing.Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _encode_float(value: float) -> typing.Any:
    if (math.isnan(value)):
        return {TAG: "float", "v": "nan"}

    if (math.isinf(value)):
        return {TAG: "float", "v": "inf" if value > 0 else "-inf"}

    return value


def _encode_series_values(values: typing.Iterable) -> list:
    return [encode(value) for value in values]


def _encode_frame(frame: pd.DataFrame) -> dict:
    columns = []

    for (position, name) in enumerate(frame.columns):
        column = frame.iloc[:, position]
        columns.append({
            "name": encode(name), "dtype": str(column.dtype), "values": _encode_series_values(column.astype(object))
        })

    if (isinstance(frame.index, pd.RangeIndex)):
        index = {
            "name": encode(frame.index.name),
            "range": [frame.index.start, frame.index.stop, frame.index.step],
        }
    else:
        index = {
            "name": encode(frame.index.name),
            "dtype": str(frame.index.dtype),
            "values": _encode_series_values(frame.index.astype(object)),
        }

    return {TAG: "frame", "columns": columns, "index": index}


def _decode_frame(state: dict) -> pd.DataFrame:
    index_state = state["index"]

    if ("range" in index_state):
        index = pd.RangeIndex(*index_state["range"], name=decode(index_state["name"]))
    else:
        index = pd.Index([decode(value) for value in index_state["values"]],
                         dtype=_series_dtype(index_state["dtype"]),
                         name=decode(index_state["name"]))
    data = {}
    names = []

    for column in state["columns"]:
        name = decode(column["name"])
        values = [decode(value) for value in column["values"]]
        names.append(name)
        data[len(names) - 1] = pd.Series(values, index=index, dtype=_series_dtype(column["dtype"]))

    frame = pd.DataFrame(data, index=index)
    frame.columns = names

    return frame


def _series_dtype(dtype: str) -> typing.Any:
    # An object column is rebuilt as object, so values pandas would otherwise coerce keep their own types.
    return object if dtype == "object" else dtype


def _encode_pairs(tag: str) -> typing.Callable[[typing.Any], dict]:
    return lambda value: {TAG: tag, "items": [[encode(k), encode(v)] for (k, v) in value.items()]}


def _encode_items(tag: str, ordered: bool = True) -> typing.Callable[[typing.Any], dict]:

    def _encode(value: typing.Any) -> dict:
        items = [encode(item) for item in value]

        # A set has no order a correct tracker may rely on, and its iteration order moves with the hash seed.
        return {TAG: tag, "items": items if ordered else sorted(items, key=_canonical)}

    return _encode


def _encode_list(value: list) -> list:
    return [encode(item) for item in value]


def _encode_deque(value: collections.deque) -> dict:
    return {TAG: "deque", "maxlen": value.maxlen, "items": [encode(item) for item in value]}


def _encode_bytes(value: typing.Union[bytes, bytearray]) -> dict:
    return {TAG: "bytes", "b64": base64.b64encode(bytes(value)).decode("ascii")}


def _encode_timestamp(value: pd.Timestamp) -> dict:
    return {TAG: "timestamp", "ns": value.value, "tz": None if value.tz is None else str(value.tz)}


def _encode_timedelta(value: pd.Timedelta) -> dict:
    return {TAG: "timedelta", "ns": value.value}


def _encode_datetime(value: datetime.datetime) -> dict:
    return {TAG: "datetime", "iso": value.isoformat()}


def _encode_ndarray(value: np.ndarray) -> dict:
    values = [encode(item) for item in value.ravel().tolist()]

    return {TAG: "ndarray", "dtype": value.dtype.str, "shape": list(value.shape), "values": values}


_ENCODERS: dict[type, typing.Callable[[typing.Any], typing.Any]] = {
    list: _encode_list,
    tuple: _encode_items("tuple"),
    set: _encode_items("set", ordered=False),
    frozenset: _encode_items("frozenset", ordered=False),
    dict: _encode_pairs("dict"),
    collections.OrderedDict: _encode_pairs("odict"),
    collections.Counter: _encode_pairs("counter"),
    collections.deque: _encode_deque,
    bytes: _encode_bytes,
    bytearray: _encode_bytes,
    pd.DataFrame: _encode_frame,
    pd.Timestamp: _encode_timestamp,
    pd.Timedelta: _encode_timedelta,
    datetime.datetime: _encode_datetime,
    np.ndarray: _encode_ndarray,
}
"""Encoders keyed on the exact type, so a subclass with behaviour of its own -- a `defaultdict`, say -- is refused
rather than silently rebuilt as its base class."""


def _encode_function(value: types.FunctionType) -> dict:
    # A module-level Morpheus function is a named choice, such as a tracker's reduction, and is rebuilt by name. A
    # lambda or a closure has no name to rebuild it by.
    if (not value.__module__.startswith(ALLOWED_MODULE_PREFIX) or "<" in value.__qualname__):
        raise StateError(f"Cannot encode {value!r}: only module-level Morpheus functions can be named in a snapshot")

    return {TAG: "func", "name": f"{value.__module__}:{value.__qualname__}"}


def _encode_object(value: typing.Any) -> dict:
    cls = type(value)

    if (isinstance(value, tuple) and hasattr(cls, "_fields")):
        return {TAG: "namedtuple", "cls": _qualified_name(cls), "items": [encode(item) for item in value]}

    if (not cls.__module__.startswith(ALLOWED_MODULE_PREFIX)):
        raise StateError(f"Cannot encode a {_qualified_name(cls)}: only Morpheus classes and plain values are "
                         f"state, and a callable or a handle is configuration")

    return {
        TAG: "obj",
        "cls": _qualified_name(cls),
        "attrs": [[name, encode(attr)] for (name, attr) in _attributes(value).items()],
    }


def encode(value: typing.Any) -> typing.Any:
    """
    Encode a value as JSON-safe plain types, keeping its type and, for mappings, its order.

    Parameters
    ----------
    value : typing.Any
        A tracker, or anything a tracker holds.

    Returns
    -------
    typing.Any
        Plain values: `None`, `bool`, `int`, `float`, `str`, lists, and dicts whose keys are strings.

    Raises
    ------
    StateError
        If the value holds something with no faithful encoding, such as a lambda or an open file.
    """
    # Order matters: an enum or a numpy scalar is also an int or a float, and bool is an int too.
    if (value is None or isinstance(value, (bool, str))):
        result = value
    elif (isinstance(value, enum.Enum)):
        result = {TAG: "enum", "cls": _qualified_name(type(value)), "v": encode(value.value)}
    elif (isinstance(value, np.generic)):
        result = {TAG: "np", "dtype": value.dtype.str, "v": encode(value.item())}
    elif (isinstance(value, int)):
        result = value
    elif (isinstance(value, float)):
        result = _encode_float(value)
    elif (value is pd.NaT or value is pd.NA):
        result = {TAG: "nat" if value is pd.NaT else "na"}
    elif (type(value) in _ENCODERS):
        result = _ENCODERS[type(value)](value)
    elif (isinstance(value, types.FunctionType)):
        result = _encode_function(value)
    else:
        result = _encode_object(value)

    return result


def _decode_timestamp(state: dict) -> pd.Timestamp:
    stamp = pd.Timestamp(state["ns"], unit="ns")

    return stamp if state["tz"] is None else stamp.tz_localize("UTC").tz_convert(state["tz"])


def _decode_ndarray(state: dict) -> np.ndarray:
    values = [decode(item) for item in state["values"]]

    return np.array(values, dtype=np.dtype(state["dtype"])).reshape(state["shape"])


def _decode_object(state: dict) -> typing.Any:
    cls = _resolve_class(state["cls"])
    obj = cls.__new__(cls)

    for (name, attr) in state["attrs"]:
        # object.__setattr__ rather than setattr, so a frozen dataclass is rebuilt as it was.
        object.__setattr__(obj, name, decode(attr))

    return obj


def _decode_pairs(state: dict) -> list[tuple]:
    return [(decode(k), decode(v)) for (k, v) in state["items"]]


def _decode_items(state: dict) -> list:
    return [decode(item) for item in state["items"]]


SPECIAL_FLOATS = {"nan": math.nan, "inf": math.inf, "-inf": -math.inf}

_DECODERS: dict[str, typing.Callable[[dict], typing.Any]] = {
    "float": lambda state: SPECIAL_FLOATS[state["v"]],
    "enum": lambda state: _resolve_class(state["cls"])(decode(state["v"])),
    "np": lambda state: np.dtype(state["dtype"]).type(decode(state["v"])),
    "nat": lambda state: pd.NaT,
    "na": lambda state: pd.NA,
    "timestamp": _decode_timestamp,
    "timedelta": lambda state: pd.Timedelta(state["ns"], unit="ns"),
    "datetime": lambda state: datetime.datetime.fromisoformat(state["iso"]),
    "bytes": lambda state: base64.b64decode(state["b64"]),
    "frame": _decode_frame,
    "ndarray": _decode_ndarray,
    "deque": lambda state: collections.deque(_decode_items(state), maxlen=state["maxlen"]),
    "counter": lambda state: collections.Counter(dict(_decode_pairs(state))),
    "odict": lambda state: collections.OrderedDict(_decode_pairs(state)),
    "dict": lambda state: dict(_decode_pairs(state)),
    "set": lambda state: set(_decode_items(state)),
    "frozenset": lambda state: frozenset(_decode_items(state)),
    "tuple": lambda state: tuple(_decode_items(state)),
    "namedtuple": lambda state: _resolve_class(state["cls"])(*_decode_items(state)),
    "func": lambda state: _resolve_attribute(state["name"]),
    "obj": _decode_object,
}


def decode(state: typing.Any) -> typing.Any:
    """
    Rebuild a value from `encode`'s output.

    Parameters
    ----------
    state : typing.Any
        What `encode` returned, or that after a JSON round trip.

    Returns
    -------
    typing.Any
        A value equal to the one encoded, of the same type.

    Raises
    ------
    StateError
        If the state names an unknown tag or a class outside the `morpheus` package.
    """
    if (state is None or isinstance(state, (bool, int, float, str))):
        return state

    if (isinstance(state, list)):
        return [decode(item) for item in state]

    if (not isinstance(state, dict) or state.get(TAG) not in _DECODERS):
        raise StateError(f"Not an encoded value: {state!r}")

    return _DECODERS[state[TAG]](state)


def to_bytes(state: typing.Any) -> bytes:
    """The canonical bytes of an encoded state: the same state always gives the same bytes."""
    return _canonical(state).encode("ascii")


def from_bytes(data: bytes) -> typing.Any:
    """Parse bytes written by `to_bytes` back into an encoded state, without decoding it."""
    return json.loads(data.decode("ascii"))


def state_digest(state: typing.Any) -> str:
    """SHA-256 of an encoded state's canonical bytes, as `sha256:<hex>`."""
    return "sha256:" + hashlib.sha256(to_bytes(state)).hexdigest()


class StateMixin:
    """
    `to_state` and `from_state` for a tracker, over everything the instance holds.

    A tracker gains the pair by inheriting this; it needs no code of its own, because the state is its attributes.
    A tracker holding something that is configuration rather than memory -- a callable, say -- lists the attribute
    in `_state_exclude` and is handed it back by `from_state`'s keyword arguments.
    """

    _state_exclude: typing.ClassVar[tuple[str, ...]] = ()

    def to_state(self) -> dict:
        """
        Everything this tracker remembers, as plain Python types.

        Returns
        -------
        dict
            `{"format", "class", "attrs"}`, JSON-safe, with entity order as the tracker holds it. Two trackers that
            have seen the same observations give byte-identical `to_bytes` of this.
        """
        attrs = [[name, encode(value)] for (name, value) in _attributes(self).items()
                 if name not in self._state_exclude]

        return {"format": STATE_FORMAT, "class": _qualified_name(type(self)), "attrs": attrs}

    @classmethod
    def from_state(cls, state: dict, **excluded: typing.Any):
        """
        Rebuild a tracker that continues exactly where the one that wrote `state` stopped.

        Parameters
        ----------
        state : dict
            What `to_state` returned, directly or after a JSON round trip.
        **excluded
            Values for the attributes this class lists in `_state_exclude`.

        Returns
        -------
        Self
            A tracker equal in every attribute to the one snapshotted.

        Raises
        ------
        StateError
            If the snapshot is of another format or another class.
        """
        if (state.get("format") != STATE_FORMAT):
            raise StateError(f"Snapshot format {state.get('format')!r} is not {STATE_FORMAT}")

        if (state.get("class") != _qualified_name(cls)):
            raise StateError(f"Snapshot is of {state.get('class')!r}, not {_qualified_name(cls)!r}")

        missing = sorted(set(cls._state_exclude) - set(excluded))

        if (len(missing) > 0):
            raise StateError(f"{cls.__name__}.from_state needs {missing}, which a snapshot does not carry")

        obj = cls.__new__(cls)

        for (name, attr) in state["attrs"]:
            object.__setattr__(obj, name, decode(attr))

        for (name, value) in excluded.items():
            object.__setattr__(obj, name, value)

        return obj
