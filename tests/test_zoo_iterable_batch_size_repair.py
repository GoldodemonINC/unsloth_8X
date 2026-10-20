# Unsloth - 2x faster, 60% less VRAM LLM training and finetuning
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.

"""The streaming-dataset NameError repair in ``unsloth/import_fixes.py``.

A partial unsloth-zoo install can call ``_iterable_batch_size`` without defining it, and the
failure only shows up at the first mapped batch of a streaming run, which reads as "streaming
works sometimes". These tests pin the helper's semantics and prove the repair fixes an actual
call site, not just the module attribute.
"""

from __future__ import annotations

import sys
import types

import pytest

from unsloth.import_fixes import (
    _ITERABLE_BATCH_SIZE_DEFAULT,
    _iterable_batch_size,
    fix_unsloth_zoo_iterable_batch_size,
)


class _ExIterable:
    def __init__(self, batch_size):
        self.batch_size = batch_size


class _Stream:
    def __init__(self, batch_size = None):
        if batch_size is not None:
            self._ex_iterable = _ExIterable(batch_size)


def _fake_zoo(monkeypatch, *, helper = None):
    """An unsloth_zoo that has dataset_utils, optionally already carrying the helper."""
    package = types.ModuleType("unsloth_zoo")
    package.__path__ = []
    dataset_utils = types.ModuleType("unsloth_zoo.dataset_utils")
    # Defined inside the fake module's namespace, exactly like the zoo call site: the name is
    # resolved from that module's globals at call time, which is what the repair rebinds.
    exec(
        "def map_stream(dataset, fn = None):\n"
        "    return _iterable_batch_size(dataset)\n",
        dataset_utils.__dict__,
    )
    if helper is not None:
        dataset_utils._iterable_batch_size = helper
    package.dataset_utils = dataset_utils
    monkeypatch.setitem(sys.modules, "unsloth_zoo", package)
    monkeypatch.setitem(sys.modules, "unsloth_zoo.dataset_utils", dataset_utils)
    return dataset_utils


def test_batch_size_defaults_when_the_stream_has_none():
    assert _iterable_batch_size(_Stream()) == _ITERABLE_BATCH_SIZE_DEFAULT
    assert _iterable_batch_size(object()) == _ITERABLE_BATCH_SIZE_DEFAULT
    # datasets' IterableDataset.map default, not an accident of getattr().
    assert _ITERABLE_BATCH_SIZE_DEFAULT == 1000


def test_batch_size_keeps_a_real_value_and_rejects_zero():
    assert _iterable_batch_size(_Stream(64)) == 64
    assert _iterable_batch_size(_Stream(0)) == _ITERABLE_BATCH_SIZE_DEFAULT
    assert _iterable_batch_size(_Stream("not a number")) == _ITERABLE_BATCH_SIZE_DEFAULT


def test_the_repair_makes_a_missing_helper_callable(monkeypatch):
    dataset_utils = _fake_zoo(monkeypatch)

    with pytest.raises(NameError, match = "_iterable_batch_size"):
        dataset_utils.map_stream(_Stream(16))

    assert fix_unsloth_zoo_iterable_batch_size() is True
    assert dataset_utils.map_stream(_Stream(16)) == 16
    assert dataset_utils.map_stream(_Stream()) == _ITERABLE_BATCH_SIZE_DEFAULT
    # Idempotent: a second import pass does not report a repair it did not make.
    assert fix_unsloth_zoo_iterable_batch_size() is False


def test_an_existing_helper_is_never_replaced(monkeypatch):
    sentinel = lambda dataset, default = 7: 7
    dataset_utils = _fake_zoo(monkeypatch, helper = sentinel)

    assert fix_unsloth_zoo_iterable_batch_size() is False
    assert dataset_utils._iterable_batch_size is sentinel
    assert dataset_utils.map_stream(_Stream(3)) == 7


def test_an_absent_zoo_is_not_an_error(monkeypatch):
    # A minimal install must still import unsloth; the repair declines quietly.
    monkeypatch.setitem(sys.modules, "unsloth_zoo", None)
    assert fix_unsloth_zoo_iterable_batch_size() is False
