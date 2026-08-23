from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd


class CsvLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.path.unlink()
        self._has_header = False

    def append(self, row: Mapping) -> None:
        pd.DataFrame([dict(row)]).to_csv(
            self.path,
            mode="a",
            header=not self._has_header,
            index=False,
            encoding="utf-8-sig",
        )
        self._has_header = True
