# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The synthetic corpora the compositions in `morpheus.compositions` are tested over.

Until plan section 4.5 each corpus and its composition were one module under `tests/morpheus/determinism/`, and
every deployment runner and tool put that directory on its path to import them (guide 13, plan item 5). The
compositions are deployable and now live in the package; the corpora are data, and live here, where the tests, the
sample and fixture generators, the clock-skew experiment, the layer 5 model training and the upstream-reuse
evaluation read them. Nothing in the package imports this directory.

`merged(name)` is the corpus and its composition as one namespace -- the shape the tests and the tools have always
used, `tp_.build_corpus()` beside `tp_.run_pipeline(...)` -- read-only: a name is looked up in the corpus first and
the composition second, and assigning to the namespace changes neither module.

**The corpus's estate is the default context there.** A composition takes the estate's context as an input and refuses
to run without it (`morpheus.compositions.estate_context`); a corpus that has an estate says so with `context()`. In
the merged namespace every composition function that takes `context` defaults it to the corpus's, so a harness that
runs a corpus through its composition need not pass it, while the composition itself, imported from the package, still
has no default to fall back on.
"""

import functools
import importlib
import inspect
import types

_MERGED: dict = {}


def merged(name: str) -> types.ModuleType:
    """The corpus `corpora.<name>` and the composition `morpheus.compositions.<name>` as one namespace."""
    if (name not in _MERGED):
        composition = importlib.import_module(f"morpheus.compositions.{name}")
        corpus = importlib.import_module(f"corpora.{name}")
        namespace = types.ModuleType(f"{name}_pipeline", composition.__doc__)

        for module in (composition, corpus):
            namespace.__dict__.update({key: value for (key, value) in vars(module).items() if not key.startswith("__")})

        if (callable(getattr(corpus, "context", None))):
            for (key, value) in vars(composition).items():
                if (inspect.isfunction(value) and value.__module__ == composition.__name__
                        and "context" in inspect.signature(value).parameters):
                    namespace.__dict__[key] = _with_context(value, corpus.context)

        namespace.__composition__ = composition
        namespace.__corpus__ = corpus
        _MERGED[name] = namespace

    return _MERGED[name]


def _with_context(function, context):
    """`function`, with its keyword-only `context` defaulting to the corpus's estate when the caller gives none."""

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        if (kwargs.get("context") is None):
            kwargs["context"] = context()

        return function(*args, **kwargs)

    return wrapper
