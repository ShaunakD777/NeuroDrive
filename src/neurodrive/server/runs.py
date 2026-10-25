"""Run folders as REST answers (api v1.10, ADR-031): the server side of the toy lab.

``RunLibrary`` reads the run folders under one directory (``runs/`` by default) and turns them into
the message models of ``schemas.py``. It only reads: training writes the folders. A run id must
look like a run folder name (``<YYYYMMDD-HHMMSS>-<method>-<tag>``, letters, digits, ``_`` and
``-``), so a request can never reach outside the runs directory.
"""

import re
from pathlib import Path

import numpy as np

from neurodrive.core.nn.network import Network
from neurodrive.experiment.store import RunStore
from neurodrive.experiment.toy_view import toy_checkpoint_view
from neurodrive.server.schemas import (
    CheckpointList,
    CheckpointNetwork,
    RunDetail,
    RunListResponse,
    RunMetrics,
    RunSummary,
    ToyCheckpointResponse,
)

RUN_ID = re.compile(r"\d{8}-\d{6}-[A-Za-z0-9_-]+")


def _sig(value: float | None, digits: int = 6) -> float | None:
    """``value`` rounded to ``digits`` significant digits (None stays None)."""
    return None if value is None else float(f"{value:.{digits}g}")


def _sig_array(values: np.ndarray, digits: int = 6) -> list:
    return np.vectorize(lambda v: float(f"{v:.{digits}g}"), otypes=[float])(values).tolist()


def _network(run_id: str, unit: int, net: Network) -> CheckpointNetwork:
    spec = net.spec
    return CheckpointNetwork(
        run_id=run_id,
        unit=unit,
        layer_sizes=list(spec.layer_sizes),
        activations=[spec.activation_name(i) for i in range(spec.n_layers)],  # type: ignore[misc]
        weights=[np.round(w, 4).tolist() for w in net.weights],
        biases=[np.round(b, 4).tolist() for b in net.biases],
    )


class RunLibrary:
    """Read-only access to the run folders in ``runs_dir`` (it may not exist yet)."""

    def __init__(self, runs_dir: Path) -> None:
        self.runs_dir = Path(runs_dir)

    def store(self, run_id: str) -> RunStore:
        """The run folder called ``run_id``; ``KeyError`` if the name is malformed or unknown."""
        if not RUN_ID.fullmatch(run_id):
            raise KeyError(run_id)
        path = self.runs_dir / run_id
        if not (path / "meta.json").is_file():
            raise KeyError(run_id)
        return RunStore.open(path)

    def list(self) -> RunListResponse:
        runs = []
        if self.runs_dir.is_dir():
            for path in sorted(self.runs_dir.iterdir(), reverse=True):  # names start with time
                if not (RUN_ID.fullmatch(path.name) and (path / "meta.json").is_file()):
                    continue
                store = RunStore.open(path)
                meta, units = store.read_meta(), store.checkpoint_units()
                runs.append(
                    RunSummary(
                        run_id=path.name,
                        method=meta["method"],
                        tag=meta["tag"],
                        status=meta["status"],
                        started_at=meta["started_at"],
                        ended_at=meta["ended_at"],
                        checkpoints=len(units),
                        last_unit=units[-1] if units else None,
                    )
                )
        return RunListResponse(runs=runs)

    def detail(self, run_id: str) -> RunDetail:
        store = self.store(run_id)
        return RunDetail(run_id=run_id, config=store.read_config(), meta=store.read_meta())

    def metrics(self, run_id: str) -> RunMetrics:
        records = self.store(run_id).read_metrics()
        layers = max((len(r.get("grad_norm") or []) for r in records), default=0)
        grad_norm = [
            [_sig((r.get("grad_norm") or [None] * layers)[layer]) for r in records]
            for layer in range(layers)
        ]
        return RunMetrics(
            run_id=run_id,
            unit=[int(r["unit"]) for r in records],
            loss=[_sig(r.get("loss")) for r in records],
            loss_before=[_sig(r.get("loss_before")) for r in records],
            grad_norm=grad_norm,
            diverged=[bool(r.get("diverged", False)) for r in records],
        )

    def checkpoints(self, run_id: str) -> CheckpointList:
        return CheckpointList(run_id=run_id, units=self.store(run_id).checkpoint_units())

    def network(self, run_id: str, unit: int) -> CheckpointNetwork:
        return _network(run_id, unit, self.store(run_id).load_checkpoint(unit))

    def toy(self, run_id: str, unit: int) -> ToyCheckpointResponse:
        """The toy lab's numbers for one checkpoint; ``ValueError`` if it is not a toy run."""
        store = self.store(run_id)
        if not store.read_meta()["method"].startswith("toy_"):
            raise ValueError(f"{run_id} is not a toy run")
        view = toy_checkpoint_view(store, unit)
        return ToyCheckpointResponse(
            run_id=run_id,
            unit=unit,
            network=_network(run_id, unit, view.network),
            loss=_sig(view.loss),
            data_x=np.round(view.data_x, 4).tolist(),
            data_y=np.round(view.data_y, 4).tolist(),
            grid_x=np.round(view.grid_x, 4).tolist(),
            grid_y=None if view.grid_y is None else np.round(view.grid_y, 4).tolist(),
            prediction_curve=(
                None
                if view.prediction_curve is None
                else np.round(view.prediction_curve, 4).tolist()
            ),
            prediction_map=(
                None if view.prediction_map is None else np.round(view.prediction_map, 4).tolist()
            ),
            weight_gradients=[_sig_array(g) for g in view.weight_gradients],
            bias_gradients=[_sig_array(g) for g in view.bias_gradients],
            glow=[np.round(g, 3).tolist() for g in view.glow],
            max_gradient=_sig(view.max_gradient),
        )
