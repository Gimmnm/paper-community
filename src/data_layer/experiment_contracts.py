from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional


AlgorithmName = Literal["leiden", "leiden_cpm", "louvain", "coarse_kmeans"]


def _as_path_str(v: Optional[str]) -> Optional[str]:
    if v is None:
        return None
    return str(Path(v))


@dataclass
class ExperimentRunManifest:
    run_id: str
    algorithm: AlgorithmName
    time_window: str  # e.g. all | 1y | 5y | y2010_2014 (calendar refit bundles)
    title: str
    leiden_dir: str
    graph_npz: str
    keyword_index_dir: Optional[str] = None
    coords_2d_path: Optional[str] = None
    topic_communities_csv: Optional[str] = None
    default_resolution: float = 0.2
    partition_type: Optional[str] = None
    tags: Dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["leiden_dir"] = _as_path_str(self.leiden_dir)
        d["graph_npz"] = _as_path_str(self.graph_npz)
        d["keyword_index_dir"] = _as_path_str(self.keyword_index_dir)
        d["coords_2d_path"] = _as_path_str(self.coords_2d_path)
        d["topic_communities_csv"] = _as_path_str(self.topic_communities_csv)
        return d

    @staticmethod
    def from_json_dict(d: Dict[str, Any]) -> "ExperimentRunManifest":
        return ExperimentRunManifest(
            run_id=str(d.get("run_id") or ""),
            algorithm=str(d.get("algorithm") or "leiden_cpm"),  # type: ignore[arg-type]
            time_window=str(d.get("time_window") or "all"),  # type: ignore[arg-type]
            title=str(d.get("title") or d.get("run_id") or "unnamed run"),
            leiden_dir=str(d.get("leiden_dir") or ""),
            graph_npz=str(d.get("graph_npz") or ""),
            keyword_index_dir=None if d.get("keyword_index_dir") in (None, "") else str(d.get("keyword_index_dir")),
            coords_2d_path=None if d.get("coords_2d_path") in (None, "") else str(d.get("coords_2d_path")),
            topic_communities_csv=(
                None if d.get("topic_communities_csv") in (None, "") else str(d.get("topic_communities_csv"))
            ),
            default_resolution=float(d.get("default_resolution", 0.2)),
            partition_type=None if d.get("partition_type") in (None, "") else str(d.get("partition_type")),
            tags=dict(d.get("tags") or {}),
        )


@dataclass
class ExperimentMetricRow:
    run_id: str
    algorithm: str
    time_window: str
    resolution_min: Optional[float]
    resolution_max: Optional[float]
    n_resolution_points: int
    mean_runtime_sec: Optional[float]
    min_runtime_sec: Optional[float]
    max_runtime_sec: Optional[float]
    mean_n_communities: Optional[float]
    max_n_communities: Optional[int]
    min_n_communities: Optional[int]
    retrieval_score: Optional[float] = None
    topic_score: Optional[float] = None
    practical_score: Optional[float] = None
    mean_runtime_active_sec: Optional[float] = None
    n_partitions_cached: Optional[int] = None
    n_partitions_computed: Optional[int] = None
    eval_sweep_plot: Optional[str] = None
    eval_breakpoints_plot: Optional[str] = None
    eval_layered_plot: Optional[str] = None

    def to_json_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentEvaluationBundle:
    generated_at_unix: float
    rows: List[ExperimentMetricRow]
    notes: List[str] = field(default_factory=list)

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "generated_at_unix": float(self.generated_at_unix),
            "rows": [x.to_json_dict() for x in self.rows],
            "notes": [str(x) for x in self.notes],
        }
