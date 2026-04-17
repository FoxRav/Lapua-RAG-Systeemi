"""Shared test fixtures."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture
def tmp_storage(tmp_path: Path) -> Iterator[Path]:
    (tmp_path / "pages").mkdir()
    (tmp_path / "tables").mkdir()
    yield tmp_path
