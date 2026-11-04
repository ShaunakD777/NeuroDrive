"""Demonstration datasets as REST answers (api v1.15, task 6.2, ADR-045): the dataset inspector.

``DatasetLibrary`` reads the dataset folders under one directory (``data/demos/`` by default,
dataset v1, ARCHITECTURE §6.16) and turns them into the message models of ``schemas.py``; the
numbers come from ``experiment/dataset_view.py``. It only reads: ``nd demo-record`` writes the
folders. A dataset id must look like a dataset folder name (``<YYYYMMDD-HHMMSS>-<teacher>-<tag>``,
letters, digits, ``_`` and ``-``), so a request can never reach outside the datasets directory.
"""

import json
import re
import zipfile
from pathlib import Path

import numpy as np

from neurodrive.experiment.dataset_view import Histogram, dataset_view
from neurodrive.learn.supervised.demos import META_FILE, DemoDataset
from neurodrive.server.schemas import (
    DatasetDetail,
    DatasetHistogram,
    DatasetListResponse,
    DatasetMap,
    DatasetSummary,
)

DATASET_ID = re.compile(r"\d{8}-\d{6}-[A-Za-z0-9_-]+")


def _histogram(h: Histogram) -> DatasetHistogram:
    return DatasetHistogram(
        name=h.name,
        lo=h.lo,
        hi=h.hi,
        edges=np.round(h.edges, 6).tolist(),
        counts=h.counts.tolist(),
        min=h.min,
        max=h.max,
        covered=round(h.covered, 6),
    )


class DatasetLibrary:
    """Read-only access to the dataset folders in ``datasets_dir`` (it may not exist yet)."""

    def __init__(self, datasets_dir: Path) -> None:
        self.datasets_dir = Path(datasets_dir)

    def _folder(self, dataset_id: str) -> Path:
        """The folder called ``dataset_id``; ``KeyError`` if the name is malformed or unknown."""
        if not DATASET_ID.fullmatch(dataset_id):
            raise KeyError(dataset_id)
        path = self.datasets_dir / dataset_id
        if not (path / META_FILE).is_file():
            raise KeyError(dataset_id)
        return path

    def list(self) -> DatasetListResponse:
        """Every folder with a readable ``meta.json``, newest first (the names start with the
        time); a folder whose metadata cannot be read is left out."""
        datasets = []
        if self.datasets_dir.is_dir():
            for path in sorted(self.datasets_dir.iterdir(), reverse=True):
                if not (DATASET_ID.fullmatch(path.name) and (path / META_FILE).is_file()):
                    continue
                try:
                    meta = json.loads((path / META_FILE).read_text(encoding="utf-8"))
                    episodes = meta["episodes"]
                    datasets.append(
                        DatasetSummary(
                            dataset_id=path.name,
                            teacher=meta["teacher"]["kind"],
                            tag=meta.get("tag", ""),
                            track=meta["track"],
                            created_at=meta.get("created_at", ""),
                            demo_frames=meta["demo_frames"],
                            episodes=len(episodes),
                            finished=sum(e["status"] == "finished" for e in episodes),
                            straight_share=meta["straight_share"],
                        )
                    )
                except (OSError, ValueError, KeyError, TypeError):
                    continue
        return DatasetListResponse(datasets=datasets)

    def detail(self, dataset_id: str) -> DatasetDetail:
        """The inspector's view of one dataset. ``KeyError`` for an unknown id, ``ValueError`` for
        a folder that does not load as dataset v1."""
        path = self._folder(dataset_id)
        try:
            dataset = DemoDataset.load(path)
        except (OSError, KeyError, zipfile.BadZipFile) as error:
            raise ValueError(f"dataset {dataset_id!r} cannot be read: {error}") from None
        view = dataset_view(dataset)
        rows = view.rows
        return DatasetDetail(
            dataset_id=dataset_id,
            meta=dataset.meta,
            labels=[_histogram(h) for h in view.labels],
            observations=[_histogram(h) for h in view.observations],
            map=DatasetMap(
                stride=view.stride,
                x=np.round(dataset.x[rows], 2).tolist(),
                y=np.round(dataset.y[rows], 2).tolist(),
                steer=np.round(dataset.label[rows, 0], 4).tolist(),
                throttle=np.round(dataset.label[rows, 1], 4).tolist(),
                episode=dataset.episode[rows].tolist(),
            ),
        )
