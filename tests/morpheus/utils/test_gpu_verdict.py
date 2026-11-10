#!/usr/bin/env python
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
A CI run says how old the GPU verdict is against the code it tests (guide 13, plan item 9).

`ci/scripts/gpu_verdict.py` reads the newest conformance artifact and counts the commits since it that changed the
GPU path. These hold it to saying the three things it can: the verdict describes this code, it does not and by how
many commits, or the history cannot tell.
"""

import importlib.util
import json
import os
import subprocess

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
MODULE_PATH = os.path.join(REPO_ROOT, "ci", "scripts", "gpu_verdict.py")


def _load():
    spec = importlib.util.spec_from_file_location("gpu_verdict", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


gpu_verdict = _load()


def _git(root, *arguments, when="2026-10-01T00:00:00+00:00"):
    environment = {**os.environ, "GIT_AUTHOR_DATE": when, "GIT_COMMITTER_DATE": when}
    subprocess.run(["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@example.com", *arguments],
                   check=True,
                   capture_output=True,
                   env=environment)


def _repository(tmp_path):
    root = tmp_path / "repo"
    (root / "ci" / "scripts").mkdir(parents=True)
    (root / "ci" / "artifacts").mkdir(parents=True)
    (root / "python" / "morpheus" / "morpheus").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "ci" / "scripts" / "gpu_conformance.sh").write_text("TARGETS=(\n    tests/test_a.py\n)\n", encoding="utf-8")
    (root / "tests" / "test_a.py").write_text("", encoding="utf-8")
    (root / "tests" / "test_b.py").write_text("", encoding="utf-8")
    (root / "python" / "morpheus" / "morpheus" / "stage.py").write_text("", encoding="utf-8")
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "start")

    return root


def _artifact(root, at, commit=None, verdict="passed"):
    report = {"verdict": verdict, "at": at, "device": "GPU"}

    if (commit is not None):
        report["commit"] = commit

    (root / "ci" / "artifacts" / "gpu_conformance-2026-10-02T0000Z.json").write_text(json.dumps(report),
                                                                                     encoding="utf-8")


def _commit(root, path, when):
    (root / path).write_text(when, encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", f"change {path}", when=when)


def test_the_gpu_path_is_the_package_and_the_tiers_test_files(tmp_path):
    root = _repository(tmp_path)

    assert gpu_verdict.gpu_path(
        str(root)) == ["python/morpheus/morpheus", "ci/scripts/gpu_conformance.sh", "tests/test_a.py"]


def test_a_verdict_no_gpu_path_commit_follows_describes_the_code(tmp_path):
    root = _repository(tmp_path)
    _artifact(root, "2026-10-02T00:00:00+00:00")
    _commit(root, "tests/test_b.py", "2026-10-03T00:00:00+00:00")

    verdict = gpu_verdict.read(str(root))
    (summary, warning) = gpu_verdict.render(verdict)

    assert (verdict.changed, verdict.current, warning) == ([], True, None)
    assert "describes this code" in summary


def test_a_gpu_path_commit_after_the_verdict_is_named(tmp_path):
    root = _repository(tmp_path)
    _artifact(root, "2026-10-02T00:00:00+00:00")
    _commit(root, "python/morpheus/morpheus/stage.py", "2026-10-03T00:00:00+00:00")
    _commit(root, "tests/test_a.py", "2026-10-04T00:00:00+00:00")

    verdict = gpu_verdict.read(str(root))
    (summary, warning) = gpu_verdict.render(verdict)

    assert [line.split(" ", 1)[1]
            for line in verdict.changed] == ["change tests/test_a.py", "change python/morpheus/morpheus/stage.py"]
    assert "2 commits since have changed the GPU path" in summary
    assert "unverified" in warning


def test_a_verdict_with_its_commit_counts_from_the_commit_not_the_clock(tmp_path):
    root = _repository(tmp_path)
    head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True,
                          check=True).stdout.strip()
    # The artifact's clock is later than the change, but its commit is earlier: the commit decides.
    _commit(root, "python/morpheus/morpheus/stage.py", "2026-10-03T00:00:00+00:00")
    _artifact(root, "2026-10-09T00:00:00+00:00", commit=head)

    assert len(gpu_verdict.read(str(root)).changed) == 1


def test_a_commit_the_history_does_not_hold_cannot_be_counted(tmp_path):
    root = _repository(tmp_path)
    _artifact(root, "2026-10-02T00:00:00+00:00", commit="0" * 40)

    verdict = gpu_verdict.read(str(root))
    (summary, warning) = gpu_verdict.render(verdict)

    assert verdict.changed is None
    assert "cannot be told" in summary and "cannot be compared" in warning


def test_no_artifact_says_no_gpu_has_run_the_code(tmp_path):
    root = _repository(tmp_path)

    (summary, warning) = gpu_verdict.render(gpu_verdict.read(str(root)))

    assert "no GPU has run this code" in summary and warning is not None


def test_a_failed_verdict_is_a_warning_even_when_current(tmp_path):
    root = _repository(tmp_path)
    _artifact(root, "2026-10-02T00:00:00+00:00", verdict="failed")

    (_, warning) = gpu_verdict.render(gpu_verdict.read(str(root)))

    assert warning is not None and "failed" in warning


def test_the_summary_is_appended_to_the_step_summary(tmp_path, monkeypatch, capsys):
    root = _repository(tmp_path)
    _artifact(root, "2026-10-02T00:00:00+00:00")
    _commit(root, "tests/test_a.py", "2026-10-03T00:00:00+00:00")
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))

    assert gpu_verdict.main(["--root", str(root)]) == 0
    assert "## GPU path" in summary.read_text(encoding="utf-8")
    assert "::warning title=GPU path::" in capsys.readouterr().out
