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
Tests for the layer 5 model runner, which is the one script in this fork that cannot run where it is tested.

`examples/layer5_model/run_model.py` trains a Torch autoencoder on a CUDA device. This environment has neither,
so what can be asserted here is everything around the model: that the runner refuses rather than reporting a
verdict it did not measure, that its refusal is recorded in the artifact rather than only on a terminal, and
that the features it would train on are ones the pipeline derived rather than raw columns a collector
sent. The model's own numbers are measured on the machine that has the card, and the artifact that run
writes is the evidence for them.
"""

import importlib.util
import json
import os
import subprocess
import sys

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RUNNER = os.path.join(REPO_ROOT, "examples", "layer5_model", "run_model.py")


def _load_runner():
    """Import the runner as a module. Nothing runs at import; `main` is guarded."""
    spec = importlib.util.spec_from_file_location("layer5_run_model", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


@pytest.fixture(name="runner_module", scope="module")
def runner_module_fixture():
    return _load_runner()


@pytest.fixture(name="corpus_and_result", scope="module")
def corpus_and_result_fixture():
    sys.path.insert(0, os.path.join(REPO_ROOT, "tests", "morpheus", "determinism"))

    import session_pipeline  # pylint: disable=import-outside-toplevel

    corpus = session_pipeline.build_corpus()

    return (corpus, session_pipeline.run_pipeline(session_pipeline.build_pipeline_config(), corpus))


def test_every_feature_is_derived_rather_than_collected(runner_module, corpus_and_result):
    # The point of the layer, stated as an assertion. A feature the collector sent -- a country, an application
    # name, an ASN -- teaches the model what the estate looks like; a feature a TC-5 stage derived teaches it
    # what this principal usually does. Adding a raw column to FEATURE_COLUMNS would quietly change which of
    # those two things gets learned, and this compares the two frames rather than trusting the naming.
    (corpus, _) = corpus_and_result
    collected = set().union(*[set(frame.columns) for frame in corpus.values()])

    assert collected, "the corpus carried no columns; the comparison would pass vacuously"

    for column in runner_module.FEATURE_COLUMNS:
        assert column not in collected, f"{column} is a column the collector sent, not one a stage derived"


def test_the_features_are_the_ones_the_corpus_actually_carries(runner_module, corpus_and_result):
    # A feature that no row ever carries is dropped by the runner's `if column in rows.columns` without a word,
    # which would silently narrow what the model trains on. Checking against the pipeline's own output means a
    # renamed or removed column fails here rather than shrinking the model.
    (_, result) = corpus_and_result

    for column in runner_module.FEATURE_COLUMNS:
        assert column in result.columns, f"{column} is in FEATURE_COLUMNS but the pipeline produces no such column"


def test_preparing_the_environment_refuses_once_torch_is_imported(runner_module):
    # `CUBLAS_WORKSPACE_CONFIG` is read when the CUDA context is created. Setting it late has no effect and
    # looks exactly like setting it correctly, so the runner refuses instead. Torch is absent here, so a stand-in
    # entry in `sys.modules` is what the guard sees -- the guard reads the name, not the package.
    sys.modules["torch"] = object()

    try:
        with pytest.raises(RuntimeError, match="imported before"):
            runner_module.prepare_environment(42)
    finally:
        del sys.modules["torch"]


def _run_with_stub_torch(tmp_path, stub: str) -> tuple:
    """Run the runner against a stand-in `torch`, so the refusal is forced rather than depended on.

    The earlier version of these tests simply ran the runner and asserted it refused, which was true only while
    the machine happened to lack Torch. Installing Torch on the machine with the card turned a passing test into
    a failing one without a line of the runner changing -- the test was asserting a property of the environment.
    A stub shadowing the real module makes both refusals reproducible anywhere, with or without a card.
    """
    stub_root = tmp_path / "stub"
    stub_root.mkdir()
    (stub_root / "torch.py").write_text(stub, encoding="utf-8")

    artifact = str(tmp_path / "layer5_model.json")
    environment = dict(os.environ, PYTHONPATH=f"{stub_root}{os.pathsep}{os.environ.get('PYTHONPATH', '')}")

    completed = subprocess.run([sys.executable, RUNNER, artifact],
                               capture_output=True,
                               text=True,
                               check=False,
                               timeout=600,
                               cwd=REPO_ROOT,
                               env=environment)

    with open(artifact, encoding="utf-8") as handle:
        return (completed, json.load(handle))


def test_the_runner_refuses_when_torch_cannot_be_imported(tmp_path):
    # The guard that matters most: a run that quietly skipped the model would report a verdict about a model it
    # never trained. It must exit non-zero, say which piece is missing, and leave a failed artifact rather than
    # no artifact -- a file that is simply absent reads afterwards as a run that was never started.
    assert os.access(RUNNER, os.X_OK), "the runner must be executable or the one command is not one command"

    (completed, report) = _run_with_stub_torch(tmp_path, "raise ImportError('stubbed out for this test')\n")

    assert completed.returncode != 0, "a machine with no Torch must not report a passing model verdict"
    assert "no Torch" in completed.stdout + completed.stderr
    assert report["verdict"] == "failed"
    assert "Torch" in report["reason"]


def test_the_runner_refuses_when_torch_reports_no_device(tmp_path):
    # The second refusal, which was never covered: Torch present and no card. The scores this measures are the
    # ones a deployment would act on, so a CPU run would answer a different question than the one asked -- and
    # answering a different question quietly is the failure this whole script is built against.
    stub = ("__version__ = '2.4.0+stub'\n"
            "class _Cuda:\n"
            "    @staticmethod\n"
            "    def is_available():\n"
            "        return False\n"
            "    @staticmethod\n"
            "    def get_device_name(index):\n"
            "        raise AssertionError('must not be asked for a device that is not available')\n"
            "cuda = _Cuda()\n")

    (completed, report) = _run_with_stub_torch(tmp_path, stub)

    assert completed.returncode != 0, "a machine with no CUDA device must not report a passing model verdict"
    assert "no CUDA device" in completed.stdout + completed.stderr
    assert report["verdict"] == "failed"


def test_the_runner_says_what_it_does_not_measure(runner_module):
    # A reproducibility verdict is easy to read as a detection verdict, and this corpus -- a week of five
    # principals -- cannot support the second. The disclaimer is part of the artifact rather than part of the
    # commentary so that it travels with the numbers it qualifies.
    assert "reproducibility, not detection quality" in runner_module.__doc__

    with open(RUNNER, encoding="utf-8") as handle:
        source = handle.read()

    assert '"measures"' in source, "the artifact must carry the caveat, not only the docstring"
