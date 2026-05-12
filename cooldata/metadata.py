from __future__ import annotations

import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import List, NamedTuple

import pandas as pd
import numpy as np

_QUAD_INDICES     = [1, 2, 3, 4]
_CYLINDER_INDICES = [5, 6]
_REPO_ID          = "datasets/bgce/cooldata-v2"


# ── Domain objects ────────────────────────────────────────────────────────────

class Position(NamedTuple):
    """Represents a 3D position. Is always the center of a body."""
    x: float
    y: float
    z: float


class Quader:
    """Represents a quader (rectangular prism) with its properties."""

    def __init__(self, temperature: float, position: Position,
                 size_x: float, size_y: float, size_z: float):
        self.temperature = temperature
        self.position    = position
        self.size_x      = size_x
        self.size_y      = size_y
        self.size_z      = size_z

    def __repr__(self) -> str:
        return (f"Quader(T={self.temperature}, Pos={self.position}, "
                f"Size=({self.size_x},{self.size_y},{self.size_z}))")


class Cylinder:
    """Represents a cylinder with its properties."""

    def __init__(self, temperature: float, position: Position,
                 radius: float, height: float):
        self.temperature = temperature
        self.position    = position
        self.radius      = radius
        self.height      = height

    def __repr__(self) -> str:
        return (f"Cylinder(T={self.temperature}, Pos={self.position}, "
                f"R={self.radius}, H={self.height})")


class SystemParameters:
    """Holds the parameters for a system of quaders, cylinders, and inflow velocity."""

    def __init__(self, quads: List[Quader], cylinders: List[Cylinder],
                 inflow_velocity: float):
        self.quads           = quads
        self.cylinders       = cylinders
        self.inflow_velocity = inflow_velocity

    def __repr__(self) -> str:
        return (f"SystemParameters(V={self.inflow_velocity}, "
                f"Quads={self.quads}, Cylinders={self.cylinders})")

    @staticmethod
    def from_dataframe_row(row: pd.Series) -> SystemParameters:
        """
        Creates a SystemParameters object from a pandas DataFrame row.
        Columns expected: T1-T6, x1-x6, y1-y6, xs1-xs4, ys1-ys4,
                          zs1-zs6, r5-r6, V.
        z-positions default to 0.0 if not present.
        """
        quads: List[Quader] = []
        for i in range(1, 5):
            quads.append(Quader(
                temperature = float(row[f"T{i}"]),
                position    = Position(float(row[f"x{i}"]), float(row[f"y{i}"]),
                                       float(row.get(f"z{i}", 0.0))),
                size_x      = float(row[f"xs{i}"]),
                size_y      = float(row[f"ys{i}"]),
                size_z      = float(row[f"zs{i}"]),
            ))

        cylinders: List[Cylinder] = []
        for i in range(5, 7):
            cylinders.append(Cylinder(
                temperature = float(row[f"T{i}"]),
                position    = Position(float(row[f"x{i}"]), float(row[f"y{i}"]),
                                       float(row.get(f"z{i}", 0.0))),
                radius      = float(row[f"r{i}"]),
                height      = float(row[f"zs{i}"]),
            ))

        return SystemParameters(quads=quads, cylinders=cylinders,
                                inflow_velocity=float(row["V"]))


def df_row_to_system_parameters(df: pd.DataFrame, design_id: int) -> SystemParameters:
    """
    Create a SystemParameters object from a DataFrame row.

    Supports two formats:
    - Trimmed metadata (has 'design_id' column): looks up by design_id value.
    - Original full metadata (no 'design_id' column): uses iloc[design_id - 1].
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame.")
    if not isinstance(design_id, int):
        raise TypeError("'design_id' must be an integer.")

    if "design_id" in df.columns:
        row = df[df["design_id"] == design_id]
        if row.empty:
            raise IndexError(f"design_id {design_id} not found in metadata.")
        return SystemParameters.from_dataframe_row(row.iloc[0])
    else:
        row_id = design_id - 1
        if not 0 <= row_id < len(df):
            raise IndexError(f"design_id {design_id} out of bounds "
                             f"(DataFrame has {len(df)} rows).")
        return SystemParameters.from_dataframe_row(df.iloc[row_id])


# ── MetadataFilter ────────────────────────────────────────────────────────────

class MetadataFilter:
    """
    Explore and download cooldata-v2 samples by metadata — no downloading needed
    until you call .load().

    Requires sample_index.json which maps design_id → run/batch for targeted
    downloads. Place it next to metadata.parquet and it loads automatically.

    Quick start
    -----------
    >>> from cooldata.metadata import MetadataFilter
    >>> f = MetadataFilter("Cooldataset/metadata.parquet")
    >>> f.summary()

    Download by filter:
    >>> ds = f.velocity(min=4.0).temperature(body=1, min=50.0).load(num_samples=50)

    Download by design ID:
    >>> ds = f.load_by_ids([125002, 125037, 212515])

    Download randomly:
    >>> ds = f.load_random(n=20)

    Download by run:
    >>> ds = f.load_by_run("run_1", num_samples=100)
    """

    def __init__(self, metadata_path: str | Path, index_path: str | Path = None):
        self._path  = Path(metadata_path)
        self._df    = pd.read_parquet(self._path)
        self._mask  = pd.Series(True, index=self._df.index)
        self._index : dict | None      = None
        self._available_ids : set[int] | None = None

        self._index_path = (Path(index_path) if index_path is not None
                            else self._path.parent / "sample_index.json")
        self._load_index()

    def _load_index(self) -> None:
        if self._index_path.exists():
            with open(self._index_path) as f:
                self._index = json.load(f)
            self._available_ids = set(int(k) for k in self._index.keys())

    def _require_index(self) -> None:
        if self._index is None:
            raise RuntimeError(
                "sample_index.json not found. Place it next to metadata.parquet."
            )

    def _get_simulated_df(self) -> pd.DataFrame:
        """Return metadata rows for simulated samples only."""
        if self._available_ids is not None:
            return self._df[self._df.index.isin(
                [did - 1 for did in self._available_ids]
            )]
        return self._df

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self) -> None:
        """
        Print a summary of the dataset.
        Scoped to simulated samples when sample_index.json is available.
        """
        if self._available_ids is not None:
            # Build a df with design_id as index for simulated rows
            sim_row_ids = [did - 1 for did in self._available_ids]
            df          = self._df.iloc[sim_row_ids]
            scope_note  = f"{len(df):,} simulated samples across 5 runs"
            match_count = self._mask.iloc[sim_row_ids].sum()
        else:
            df          = self._df
            scope_note  = (f"{len(df):,} total parameter combinations  "
                           "— place sample_index.json next to metadata.parquet "
                           "to scope to simulated samples only")
            match_count = self._mask.sum()

        # Run breakdown
        run_counts = {}
        if self._index is not None:
            for v in self._index.values():
                run_counts[v["run"]] = run_counts.get(v["run"], 0) + 1

        sep = "=" * 54
        print(sep)
        print("  Cooldata v2 — Dataset Summary")
        print(sep)
        print(f"  Showing          : {scope_note}")
        if run_counts:
            print()
            print("  Samples per run  :")
            for run, count in sorted(run_counts.items()):
                print(f"    {run} : {count:,}")
        print()
        print(f"  Inlet velocity V : {df['V'].min():.2f} – {df['V'].max():.2f}"
              f"  (mean {df['V'].mean():.2f})")
        print()
        print("  Temperatures (T1–T6):")
        for i in range(1, 7):
            col = f"T{i}"
            print(f"    T{i}: {df[col].min():.1f} – {df[col].max():.1f}"
                  f"  (mean {df[col].mean():.1f})")
        print()
        print("  Active bodies (y == 1.0 = inactive sentinel):")
        print(f"    {'Body':<8} {'Type':<12} {'Active rows':<16} {'Always active'}")
        for i in range(1, 7):
            btype    = "Quad" if i <= 4 else "Cylinder"
            n_active = (df[f"y{i}"] != 1.0).sum()
            always   = "yes" if n_active == len(df) else "no"
            print(f"    body {i:<3} {btype:<12} {n_active:>8,} / {len(df):,}   {always}")
        print()
        print("  Body positions (active only, x / y range):")
        for i in range(1, 7):
            active = df[df[f"y{i}"] != 1.0]
            if len(active) == 0:
                continue
            print(f"    body {i}: "
                  f"x [{active[f'x{i}'].min():.3f} – {active[f'x{i}'].max():.3f}]  "
                  f"y [{active[f'y{i}'].min():.3f} – {active[f'y{i}'].max():.3f}]")
        print()
        print(sep)
        print(f"  Current filter matches : {match_count:,} samples")
        print(sep)

    # ── Filter methods (chainable) ────────────────────────────────────────────

    def velocity(self, min: float = None, max: float = None) -> MetadataFilter:
        """Filter by inlet velocity V (range: 1.0 – 7.0)."""
        if min is not None:
            self._mask &= self._df["V"] >= min
        if max is not None:
            self._mask &= self._df["V"] <= max
        return self

    def temperature(self, body: int = None,
                    min: float = None, max: float = None) -> MetadataFilter:
        """
        Filter by temperature (range: 20.0 – 80.0).
        body=None : at least one body must satisfy the range.
        body=1–6  : only that specific body is checked.
        """
        if body is not None:
            if body not in range(1, 7):
                raise ValueError(f"body must be 1–6, got {body}")
            if min is not None:
                self._mask &= self._df[f"T{body}"] >= min
            if max is not None:
                self._mask &= self._df[f"T{body}"] <= max
        else:
            any_match = pd.Series(False, index=self._df.index)
            for i in range(1, 7):
                ok = pd.Series(True, index=self._df.index)
                if min is not None:
                    ok &= self._df[f"T{i}"] >= min
                if max is not None:
                    ok &= self._df[f"T{i}"] <= max
                any_match |= ok
            self._mask &= any_match
        return self

    def position(self, body: int,
                 x_min: float = None, x_max: float = None,
                 y_min: float = None, y_max: float = None) -> MetadataFilter:
        """Filter by position of a specific body (1–6)."""
        if body not in range(1, 7):
            raise ValueError(f"body must be 1–6, got {body}")
        if x_min is not None:
            self._mask &= self._df[f"x{body}"] >= x_min
        if x_max is not None:
            self._mask &= self._df[f"x{body}"] <= x_max
        if y_min is not None:
            self._mask &= self._df[f"y{body}"] >= y_min
        if y_max is not None:
            self._mask &= self._df[f"y{body}"] <= y_max
        return self

    def size(self, quad: int,
             xs_min: float = None, xs_max: float = None,
             ys_min: float = None, ys_max: float = None) -> MetadataFilter:
        """Filter by x/y size of a quad body (1–4)."""
        if quad not in _QUAD_INDICES:
            raise ValueError(f"quad must be 1–4, got {quad}")
        if xs_min is not None:
            self._mask &= self._df[f"xs{quad}"] >= xs_min
        if xs_max is not None:
            self._mask &= self._df[f"xs{quad}"] <= xs_max
        if ys_min is not None:
            self._mask &= self._df[f"ys{quad}"] >= ys_min
        if ys_max is not None:
            self._mask &= self._df[f"ys{quad}"] <= ys_max
        return self

    def radius(self, cylinder: int,
               min: float = None, max: float = None) -> MetadataFilter:
        """Filter by radius of a cylinder body (5 or 6)."""
        if cylinder not in _CYLINDER_INDICES:
            raise ValueError(f"cylinder must be 5 or 6, got {cylinder}")
        if min is not None:
            self._mask &= self._df[f"r{cylinder}"] >= min
        if max is not None:
            self._mask &= self._df[f"r{cylinder}"] <= max
        return self

    def n_quads(self, exactly: int = None,
                min: int = None, max: int = None) -> MetadataFilter:
        """
        Filter by number of active quads (bodies 1–4).
        A body is inactive when y == 1.0.
        """
        active_quads = sum(
            (self._df[f"y{i}"] != 1.0).astype(int)
            for i in _QUAD_INDICES
        )
        if exactly is not None:
            self._mask &= active_quads == exactly
        else:
            if min is not None:
                self._mask &= active_quads >= min
            if max is not None:
                self._mask &= active_quads <= max
        return self

    def n_cylinders(self, exactly: int = None,
                    min: int = None, max: int = None) -> MetadataFilter:
        """
        Filter by number of active cylinders (bodies 5–6).
        A body is inactive when y == 1.0 (sentinel value).
        """
        active_cyls = sum(
            (self._df[f"y{i}"] != 1.0).astype(int)
            for i in _CYLINDER_INDICES
        )
        if exactly is not None:
            self._mask &= active_cyls == exactly
        else:
            if min is not None:
                self._mask &= active_cyls >= min
            if max is not None:
                self._mask &= active_cyls <= max
        return self

    def n_bodies(self, exactly: int = None,
                 min: int = None, max: int = None) -> MetadataFilter:
        """
        Filter by total number of active bodies (quads + cylinders).
        A body is inactive when y == 1.0.
        """
        active = sum(
            (self._df[f"y{i}"] != 1.0).astype(int)
            for i in range(1, 7)
        )
        if exactly is not None:
            self._mask &= active == exactly
        else:
            if min is not None:
                self._mask &= active >= min
            if max is not None:
                self._mask &= active <= max
        return self

    def run(self, *run_names: str) -> MetadataFilter:
        """
        Restrict to samples from specific runs.
        Example: f.run("run_1", "run_3")
        Available runs: run_1, run_3, run_4, run_6, run_7
        """
        self._require_index()
        allowed_ids = {
            int(did) for did, entry in self._index.items()
            if entry["run"] in run_names
        }
        # Convert design IDs to row indices (design_id - 1)
        allowed_rows = {did - 1 for did in allowed_ids}
        self._mask &= self._df.index.isin(allowed_rows)
        return self

    def custom(self, expr: str) -> MetadataFilter:
        """
        Apply a raw pandas query string on the metadata columns.
        Available: T1–T6, x1–x6, y1–y6, xs1–xs4, ys1–ys4, zs1–zs6, r5, r6, V

        Example: f.custom("V > 4.5 and T1 < 80")
        """
        self._mask &= self._df.index.isin(self._df.query(expr).index)
        return self

    def simulated_only(self) -> MetadataFilter:
        """Restrict filter to design IDs that have simulation files."""
        self._require_index()
        sim_row_ids = {did - 1 for did in self._available_ids}
        self._mask &= self._df.index.isin(sim_row_ids)
        return self

    def reset(self) -> MetadataFilter:
        """Clear all filters."""
        self._mask = pd.Series(True, index=self._df.index)
        return self

    # ── Results ───────────────────────────────────────────────────────────────

    def count(self) -> int:
        """Number of samples matching the current filter (intersected with simulated)."""
        if self._available_ids is not None:
            sim_rows = {did - 1 for did in self._available_ids}
            return int((self._mask & self._df.index.isin(sim_rows)).sum())
        return int(self._mask.sum())

    def get_design_ids(self) -> list[int]:
        """
        Return design IDs matching the current filter, restricted to
        simulated samples when sample_index.json is available.
        """
        if self._available_ids is not None:
            sim_rows = {did - 1 for did in self._available_ids}
            mask = self._mask & self._df.index.isin(sim_rows)
        else:
            mask = self._mask
        # Convert row indices back to design IDs (row_index + 1)
        return [idx + 1 for idx in self._df[mask].index.tolist()]

    def get_dataframe(self) -> pd.DataFrame:
        """
        Return filtered metadata as a DataFrame with design_id as first column.
        Restricted to simulated samples when sample_index.json is available.
        """
        design_ids = self.get_design_ids()
        row_ids    = [did - 1 for did in design_ids]
        df         = self._df.iloc[row_ids].copy()
        df.insert(0, "design_id", design_ids)
        df = df.reset_index(drop=True)
        return df

    # ── Download methods ──────────────────────────────────────────────────────

    def load(self, num_samples: int = None,
             data_dir: str | Path = "Cooldataset"):
        """
        Download samples matching the current filter.

        Parameters
        ----------
        num_samples : cap on number of samples. Downloads all matches if None.
        data_dir    : local directory for downloaded files.
        """
        design_ids = self.get_design_ids()
        if num_samples is not None:
            design_ids = design_ids[:num_samples]
        return self._download(design_ids, data_dir)

    def load_by_ids(self, design_ids: list[int],
                    data_dir: str | Path = "Cooldataset"):
        """
        Download specific samples by design ID.

        Example
        -------
        >>> ds = f.load_by_ids([125002, 125037, 212515])
        """
        return self._download(design_ids, data_dir)

    def load_random(self, n: int, data_dir: str | Path = "Cooldataset",
                    seed: int = None):
        """
        Download n randomly selected samples from the simulated dataset
        (respects any active filters).

        Parameters
        ----------
        n    : number of samples to download.
        seed : optional random seed for reproducibility.
        """
        design_ids = self.get_design_ids()
        if seed is not None:
            random.seed(seed)
        chosen = random.sample(design_ids, min(n, len(design_ids)))
        print(f"Randomly selected {len(chosen)} samples (seed={seed}).")
        return self._download(chosen, data_dir)

    def load_by_run(self, run_name: str, num_samples: int = None,
                    data_dir: str | Path = "Cooldataset"):
        """
        Download samples from a specific run.
        Available runs: run_1, run_3, run_4, run_6, run_7

        Parameters
        ----------
        run_name    : e.g. "run_1"
        num_samples : optional cap.
        """
        self._require_index()
        design_ids = sorted(
            int(did) for did, entry in self._index.items()
            if entry["run"] == run_name
        )
        if not design_ids:
            raise ValueError(f"No samples found for run '{run_name}'. "
                             f"Available: {sorted(set(v['run'] for v in self._index.values()))}")
        if num_samples is not None:
            design_ids = design_ids[:num_samples]
        print(f"Downloading {len(design_ids)} samples from {run_name}.")
        return self._download(design_ids, data_dir)

    # ── Core download logic ───────────────────────────────────────────────────

    def _download(self, design_ids: list[int],
                  data_dir: str | Path = "Cooldataset"):
        """
        Download and load the given design IDs using sample_index.json
        for targeted batch downloads.
        """
        import huggingface_hub as hf
        from cooldata.pyvista_flow_field_dataset import PyvistaFlowFieldDataset, PyvistaSample

        self._require_index()

        if not design_ids:
            print("No samples to download.")
            return None

        print(f"Downloading {len(design_ids)} samples ...")

        data_dir    = Path(os.path.abspath(data_dir))
        volume_dir  = data_dir / "volume"
        surface_dir = data_dir / "surface"
        tmp_dir     = data_dir / "tmp"
        for d in (volume_dir, surface_dir, tmp_dir):
            os.makedirs(d, exist_ok=True)

        # Group by (run, batch) — only download each batch once
        grouped : dict[tuple, list[int]] = defaultdict(list)
        missing = []
        for did in design_ids:
            entry = self._index.get(str(did))
            if entry is None:
                missing.append(did)
            else:
                grouped[(entry["run"], entry["batch"])].append(did)

        if missing:
            print(f"  Warning: {len(missing)} design ID(s) not in index: "
                  f"{missing[:5]}{'...' if len(missing) > 5 else ''}")

        fs      = hf.HfFileSystem()
        samples : list[PyvistaSample] = []

        total_batches = len(grouped)
        for batch_num, ((run_name, batch_name), batch_ids) in enumerate(grouped.items(), 1):
            zip_remote  = f"{_REPO_ID}/runs/{run_name}/{batch_name}.zip"
            zip_local   = tmp_dir / run_name / f"{batch_name}.zip"
            extract_dir = tmp_dir / run_name / batch_name
            os.makedirs(zip_local.parent, exist_ok=True)

            print(f"  [{batch_num}/{total_batches}] {run_name}/{batch_name}.zip "
                  f"({len(batch_ids)} needed) ...", end=" ", flush=True)
            try:
                fs.download(zip_remote, str(zip_local))
            except Exception as e:
                print(f"FAILED: {e}")
                continue

            extract_dir.mkdir(parents=True, exist_ok=True)
            shutil.unpack_archive(str(zip_local), str(extract_dir))

            found = 0
            for did in batch_ids:
                fid = f"{did:07d}"
                v   = extract_dir / f"volume_design_{fid}_p.cgns"
                s   = extract_dir / f"surface_design_{fid}_p.cgns"
                if not v.exists() or not s.exists():
                    print(f"\n    Warning: files missing for design {fid}")
                    continue
                shutil.copy(v, volume_dir / v.name)
                shutil.copy(s, surface_dir / s.name)
                samples.append(PyvistaSample(volume_dir / v.name,
                                             surface_dir / s.name))
                found += 1

            print(f"{found}/{len(batch_ids)} copied")
            shutil.rmtree(extract_dir, ignore_errors=True)
            zip_local.unlink(missing_ok=True)

        shutil.rmtree(tmp_dir, ignore_errors=True)

        if not samples:
            print("No samples collected.")
            return None

        ds = PyvistaFlowFieldDataset(samples)
        ds.add_metadata(pd.read_parquet(self._path))
        print(f"\nLoaded {len(ds)} samples.")
        return ds
