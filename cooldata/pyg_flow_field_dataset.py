import json
import os
from pathlib import Path
from typing import Callable, List, Optional
import torch_geometric as PyG
from torch_geometric.data import Data as PyGGraph

import concurrent.futures
import shutil

import numpy as np
import pyvista as pv
import torch
from tqdm import tqdm

from cooldata.pyvista_flow_field_dataset import (
    PyvistaFlowFieldDataset,
    PyvistaSample,
    SurfaceFieldType,
)


def process_sample(args):
    cache_dir, sample, i = args
    g = PyGVolumeFlowFieldDataset.pyvista_to_volume_PyG(sample)
    print("Saving graph to cache directory:", os.path.join(cache_dir, f"{i}.pt"))
    
    torch.save(g, os.path.join(cache_dir, f"{i}.pt"))
    print("Graph saved successfully.")


class PyGVolumeFlowFieldDataset(torch.utils.data.Dataset):
    def __init__(
        self, cache_dir: str, pyvista_dataset: PyvistaFlowFieldDataset | None = None, parallel_conversion: bool = True
    ):
        """
        Creates a new PyGVolumeFlowFieldDataset. If a PyvistaFlowFieldDataset is provided, it will be converted to PyGGraphs and stored in the cache directory. If not, the dataset will be loaded from the cache directory.
        Parameters:
        -----------
        cache_dir: str
            The directory where the dataset converted to PyGGraphs is stored.
        polydata: pv.PolyData
            The directory where the dataset converted to PyGGraphs is stored. Default None.
        """
        self.cache_dir = Path(os.path.abspath(cache_dir))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.node_stats: tuple[dict, dict] | None = None
        self.edge_stats: tuple[dict, dict] | None = None
        if not self.cache_dir.exists():
            self.cache_dir.mkdir(parents=True)
        existing_files = [f for f in self.cache_dir.iterdir() if f.suffix == ".pt"]
        if len(existing_files) == len(pyvista_dataset or []):
            self.files = existing_files
            return
        if pyvista_dataset is not None:
            # clear the cache directory
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True)
            pyvista_dataset.unload()
            if parallel_conversion:
                args = [
                    (self.cache_dir, pyvista_dataset[i], i)
                    for i in range(len(pyvista_dataset))
                ]
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    list(
                        tqdm(
                            executor.map(process_sample, args),
                            total=len(pyvista_dataset),
                            desc="Converting Pyvista dataset to PyGGraphs",
                        )
                    )
            else:
                for i in tqdm(
                    range(len(pyvista_dataset)),
                    desc="Converting Pyvista dataset to PyG Graphs",
                ):
                    process_sample((self.cache_dir, pyvista_dataset[i], i))
        self.files = [f for f in self.cache_dir.iterdir() if f.suffix == ".pt"]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx) -> PyGGraph:
        
        filename = os.path.join(self.cache_dir, self.files[idx])
        if filename.endswith(".pt"):
            g: PyGGraph = torch.load(filename)
            if not isinstance(g, PyGGraph):
                raise ValueError(
                    f"File {filename} does not contain a PyGGraph object."
                )
            if self.node_stats is not None and self.edge_stats is not None:
                self.normalize_inplace(g)
            return g
        else:
            raise ValueError(f"File {filename} is not a torch file.")

    @classmethod
    def pyvista_to_volume_PyG(cls, sample: PyvistaSample) -> PyGGraph:
        """
        Convert a Pyvista UnstructuredGrid object to a PyGGraph of the volume flow field.

        Parameters:
        -----------
        grid: pv.UnstructuredGrid
            The UnstructuredGrid object to convert.

        Returns:
        --------
        PyGGraph: The converted graph.
        """

        grid = sample.volume_data[0][0][0]
        edges_from = []
        edges_to = []

        # TODO: Speed up this loop
        for i in tqdm(range(grid.n_cells)):
            neighbors = grid.cell_neighbors(i)
            edges_from.extend([i] * len(neighbors))
            edges_to.extend(neighbors)
        edge_index = torch.tensor([edges_from, edges_to], dtype=torch.long)
        graph = PyGGraph(edge_index=edge_index)
        # TODO add the attributes to the nodes from the grid
        graph["Pressure"] = torch.tensor(
            grid.cell_data["Pressure"], dtype=torch.float32
        )
        graph["Temperature"] = torch.tensor(
            grid.cell_data["Temperature"], dtype=torch.float32
        )
        centers = torch.tensor(grid.cell_centers().points, dtype=torch.float32)
        graph.pos = centers
        velocities = torch.stack(
            [
                torch.tensor(grid.cell_data["Velocity_0"], dtype=torch.float32),
                torch.tensor(grid.cell_data["Velocity_1"], dtype=torch.float32),
                torch.tensor(grid.cell_data["Velocity_2"], dtype=torch.float32),
            ],
            dim=1,
        )
        graph["Velocity"] = velocities
        graph["TurbulentDissipationRate"] = torch.tensor(
            grid.cell_data["TurbulentDissipationRate"], dtype=torch.float32
        )
        graph["TurbulentKineticEnergy"] = torch.tensor(
            grid.cell_data["TurbulentKineticEnergy"], dtype=torch.float32
        )
        graph["Volume"] = torch.tensor(
            grid.cell_data["Volume"], dtype=torch.float32
        )
        connectivity_vectors = (
            graph.pos[edges_to] - graph.pos[edges_from]
        )
        graph.edge_attr = connectivity_vectors
        return graph

    def volume_PyG_to_pv(self, graph: PyGGraph) -> pv.PolyData:
        """
        Convert a PyGGraph of the volume flow field to a Pyvista PolyData object.

        Parameters:
        -----------
        graph: PyG.PyGGraph
            The PyGGraph to convert.

        Returns:
        --------
        pv.PolyData: The converted PolyData object.
        """
        raise NotImplementedError("Implement this method")

    def normalize(self):
        if os.path.exists(os.path.join(self.cache_dir, "stats.json")):
            with open(os.path.join(self.cache_dir, "stats.json"), "r") as f:
                stats = json.load(f)
                if "node_stats" in stats and "edge_stats" in stats:
                    self.node_stats = stats["node_stats"]
                    self.edge_stats = stats["edge_stats"]
        else:
            self.node_stats = self.compute_node_stats()
            self.edge_stats = self.compute_edge_stats()
            with open(os.path.join(self.cache_dir, "stats.json"), "w") as f:
                json.dump(
                    {
                        "node_stats": self.node_stats,
                        "edge_stats": self.edge_stats,
                    },
                    f,
                )

    def denormalize(self):
        self.node_stats = None
        self.edge_stats = None

    def normalize_inplace(self, graph: PyGGraph):
        """
        Normalize the features of the graph.

        Parameters:
        -----------
        graph: PyGGraph
            The graph to normalize.
        """
        device = graph.pos.device
        if self.node_stats is None or self.edge_stats is None:
            raise ValueError(
                "Normalization statistics not computed. Call `normalize()` first."
            )
        for key in graph.keys():
            if key not in self.node_stats[0] or key not in self.node_stats[1]:
                continue
            if graph[key].dtype != torch.float32:
                continue
            graph[key] = (
                graph[key]
                - torch.tensor(self.node_stats[0][key], device=device)
            ) / torch.tensor(self.node_stats[1][key], device=device)

        for key in graph.edge_attrs():
            if key not in self.edge_stats[0] or key not in self.edge_stats[1]:
                continue
            if graph[key].dtype != torch.float32:
                continue
            graph[key] = (
                graph[key]
                - torch.tensor(self.edge_stats[0][key], device=device)
            ) / torch.tensor(self.edge_stats[1][key], device=device)

    def denormalize_inplace(self, graph: PyGGraph):
        """
        Denormalize the features of the graph.

        Parameters:
        -----------
        graph: PyG.PyGGraph
            The graph to denormalize.
        """
        if self.node_stats is None or self.edge_stats is None:
            raise ValueError(
                "Denormalization statistics not computed. Call `normalize()` first."
            )
        for key in graph.keys():
            if key not in self.node_stats:
                continue
            if graph[key].dtype != torch.float32:
                continue
            graph[key] = (
                graph[key]
                * torch.tensor(self.node_stats[key][1], device=graph.device)
            ) + torch.tensor(self.node_stats[key][0], device=graph.device)
        for key in graph.edge_attrs():
            if  key not in self.edge_stats:
                continue
            if graph[key].dtype != torch.float32:
                continue
            graph[key] = (
                graph[key]
                * torch.tensor(self.edge_stats[key][1], device=graph.device)
            ) + torch.tensor(self.edge_stats[key][0], device=graph.device)

    def compute_node_stats(self) -> tuple[dict[str, float], dict[str, float]]:
        """
        Compute the mean and standard deviation of the node features.
        """
        graph_means: dict[str, list[float]] = {}
        graph_stds: dict[str, list[float]] = {}
        for i in range(len(self)):
            graph = self[i]
            for key in graph.node_attrs():
                if graph[key].dtype != torch.float32:
                    continue
                if key not in graph_means:
                    graph_means[key] = []
                    graph_stds[key] = []
                graph_means[key].append(graph[key].mean(dim=0).tolist())
                graph_stds[key].append(graph[key].std(dim=0).tolist())
        means = {
            key: (np.sum(graph_means[key], axis=0) / len(graph_means[key])).tolist()
            for key in graph_means
        }
        stds = {
            key: (
                np.sqrt(
                    np.sum(np.square(graph_stds[key]), axis=0) / len(graph_stds[key])
                )
            ).tolist()
            for key in graph_stds
        }
        return means, stds

    def compute_edge_stats(self) -> tuple[dict[str, float], dict[str, float]]:
        """
        Compute the mean and standard deviation of the edge features.
        """
        graph_means: dict[str, list[float]] = {}
        graph_stds: dict[str, list[float]] = {}
        for i in range(len(self)):
            graph = self[i]
            for key in graph.edge_attrs():
                if graph[key].dtype != torch.float32:
                    continue
                if key not in graph_means:
                    graph_means[key] = []
                    graph_stds[key] = []
                graph_means[key].append(graph[key].mean(dim=0).tolist())
                graph_stds[key].append(graph[key].std(dim=0).tolist())
        means = {
            key: (np.sum(graph_means[key], axis=0) / len(graph_means[key])).tolist()
            for key in graph_means
        }
        stds = {
            key: (
                np.sqrt(
                    np.sum(np.square(graph_stds[key]), axis=0) / len(graph_stds[key])
                )
            ).tolist()
            for key in graph_stds
        }
        return means, stds

    @classmethod
    def l2_loss(cls, graph1: PyGGraph, graph2: PyGGraph):
        """
        Compute the L2 loss between two PyGGraphs.

        Parameters:
        -----------
        graph1: PyGGraph
            The first graph.
        graph2: PyGGraph
            The second graph.

        Returns:
        --------
        float: The L2 loss between the two graphs.
        """
        raise NotImplementedError("Implement this method")


def process_surface_sample(args):
    cache_dir, sample, i, patches_to_include = args
    g = PyGSurfaceFlowFieldDataset.pyvista_to_surface_PyG(sample, patches_to_include)
    torch.save(g, os.path.join(cache_dir, f"{i}.pt"))


class PyGSurfaceFlowFieldDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        cache_dir: str | Path,
        pyvista_dataset: PyvistaFlowFieldDataset | None = None,
        normalize: bool = True,
        patches_to_include: Optional[List[int]] = None,
    ):
        """Dataset of surface flow fields represented as PyGGraphs.

        If a PyvistaFlowFieldDataset is provided, it will be converted to PyGGraphs and stored in the cache directory. If not, the dataset will be loaded from the cache directory.

        Parameters:
        -----------
        cache_dir: str
            The directory where the dataset converted to PyGGraphs is stored.
        polydata: pv.PolyData
            The directory where the dataset converted to PyGGraphs is stored. Default None.
        """
        self.cache_dir = Path(os.path.abspath(cache_dir))
        if not self.cache_dir.exists():
            os.makedirs(self.cache_dir)
        existing = [f for f in os.listdir(self.cache_dir) if f.endswith(".pt")]
        if pyvista_dataset is not None and len(existing) != len(pyvista_dataset):
            # clear the cache directory
            for f in os.listdir(self.cache_dir):
                os.remove(os.path.join(self.cache_dir, f))
            args = [
                (self.cache_dir, pyvista_dataset[i], i, patches_to_include)
                for i in range(len(pyvista_dataset))
            ]
            with concurrent.futures.ProcessPoolExecutor() as executor:
                list(
                    tqdm(
                        executor.map(process_surface_sample, args),
                        total=len(pyvista_dataset),
                        desc="Converting Pyvista dataset to PyGGraphs (surface)",
                    )
                )
        self.files = [f for f in os.listdir(self.cache_dir) if f.endswith(".pt")]

        # first do not normalize to compute the stats, then normalize
        self.do_normalization = False
        if pyvista_dataset is not None:
            node_stats = self.compute_node_stats()
            edge_stats = self.compute_edge_stats()
            with open(os.path.join(self.cache_dir, "stats.json"), "w") as f:
                json.dump({"node_stats": node_stats, "edge_stats": edge_stats}, f)
        with open(os.path.join(self.cache_dir, "stats.json"), "r") as f:
            stats = json.load(f)
            node_stats: tuple[dict[str, float], dict[str, float]] = stats["node_stats"]
            edge_stats: tuple[dict[str, float], dict[str, float]] = stats["edge_stats"]
            self.node_means, self.node_stds = node_stats
            self.edge_means, self.edge_stds = edge_stats
        self.do_normalization = normalize

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """
        Get the graph at the given index."""
        filename = os.path.join(self.cache_dir, self.files[idx])
        g = torch.load(filename)
        if not isinstance(g, PyGGraph):
            raise ValueError(
                f"File {filename} does not contain a PyGGraph object."
            )
        if self.do_normalization:
            g = self.normalize(g)
        return g

    def slice(self, start: int, end: int) -> "PyGSurfaceFlowFieldDataset":
        """
        Get a slice of the dataset.
        """
        new_dataset = PyGSurfaceFlowFieldDataset(
            cache_dir=self.cache_dir,
            pyvista_dataset=None,
            normalize=self.do_normalization,
        )
        new_dataset.files = self.files[start:end]
        new_dataset.node_means = self.node_means
        new_dataset.node_stds = self.node_stds
        new_dataset.edge_means = self.edge_means
        new_dataset.edge_stds = self.edge_stds
        return new_dataset

    def shuffle(self) -> None:
        """
        Shuffle the dataset.
        """
        np.random.shuffle(self.files)

    def select_subset(self, indices: List[int]) -> "PyGSurfaceFlowFieldDataset":
        """
        Select a subset of the dataset based on the given indices.
        """
        new_dataset = PyGSurfaceFlowFieldDataset(
            cache_dir=self.cache_dir,
            pyvista_dataset=None,
            normalize=self.do_normalization,
        )
        new_dataset.files = [self.files[i] for i in indices]
        new_dataset.node_means = self.node_means
        new_dataset.node_stds = self.node_stds
        new_dataset.edge_means = self.edge_means
        new_dataset.edge_stds = self.edge_stds
        return new_dataset

    @classmethod
    def pyvista_to_surface_PyG(
        cls, sample: PyvistaSample, block_indices: Optional[List[int]] = None
    ) -> PyGGraph:
        """
        Convert a Pyvista UnstructuredGrid object to a PyGGraph of the volume flow field.

        Parameters:
        -----------
        grid: pv.UnstructuredGrid
            The UnstructuredGrid object to convert.

        Returns:
        --------
        PyGGraph: The converted graph.
        """
        if block_indices is None:
            block_indices = list(range(len(sample.surface_data[0])))

        for block_index in block_indices:
            grid = sample.surface_data[0][block_index]
            # TODO: Speed up this loop
            grid.cell_data["BodyID"] = block_index
            block_name = sample.surface_data[0].get_block_name(block_index)
            surface_type = -1
            if "wall" in block_name.lower():
                surface_type = 0
            elif "inlet" in block_name.lower():
                surface_type = 1
            elif "outlet" in block_name.lower():
                surface_type = 2
            elif "symmetry" in block_name.lower():
                surface_type = 3
            elif "body" in block_name.lower():
                surface_type = 4
            else:
                print(f"Unknown surface type for block {block_index}: {block_name}")
            grid.cell_data["SurfaceType"] = surface_type
            if "WallShearStress_0" not in grid.cell_data:
                grid.cell_data["WallShearStress_0"] = 0.0
                grid.cell_data["WallShearStress_1"] = 0.0
                grid.cell_data["WallShearStress_2"] = 0.0
        combined = sample.surface_data[0].combine(merge_points=True)
        edges_from = []
        edges_to = []
        for i in range(combined.n_cells):
            neighbors = combined.cell_neighbors(i)
            edges_from.extend([i] * len(neighbors))
            edges_to.extend(neighbors)
        edge_index = torch.tensor([edges_from, edges_to], dtype=torch.long)
        graph = PyGGraph(edge_index=edge_index)
        graph["Pressure"] = torch.tensor(
            combined.cell_data["Pressure"], dtype=torch.float32
        )
        graph["Temperature"] = torch.tensor(
            combined.cell_data["Temperature"], dtype=torch.float32
        )
        centers = torch.tensor(combined.cell_centers().points, dtype=torch.float32)
        graph.pos = centers
        shear_stresses = (
            torch.tensor(combined.cell_data["WallShearStress_0"], dtype=torch.float32),
            torch.tensor(combined.cell_data["WallShearStress_1"], dtype=torch.float32),
            torch.tensor(combined.cell_data["WallShearStress_2"], dtype=torch.float32),
        )

        shear_stress = torch.stack(shear_stresses, dim=1)
        graph["ShearStress"] = shear_stress
        graph["Normal"] = torch.tensor(
            combined.extract_surface().face_normals, dtype=torch.float32
        )
        graph["CellArea"] = torch.tensor(
            combined.compute_cell_sizes(
                length=False, area=True, volume=False
            ).cell_data["Area"],
            dtype=torch.float32,
        )
        graph["SurfaceType"] = torch.tensor(
            combined.cell_data["SurfaceType"], dtype=torch.int32
        )
        graph["BodyID"] = torch.tensor(
            combined.cell_data["BodyID"], dtype=torch.int32
        )
        graph["HeatTransferCoefficient"] = torch.tensor(
            combined.cell_data["HeatTransferCoefficient"], dtype=torch.float32
        )
        connectivity_vectors = (
            graph.pos[edges_to] - graph.pos[edges_from]
        )
        graph.edge_attr = connectivity_vectors
        # graph['velocity'] = torch.tensor(grid.point_data['Velocity'], dtype=torch.float32)
        # TODO add the attributes to the nodes from the grid
        return graph

    def compute_node_stats(self) -> tuple[dict[str, float], dict[str, float]]:
        """
        Compute the mean and standard deviation of the node features.
        """
        graph_means: dict[str, list[float]] = {}
        graph_stds: dict[str, list[float]] = {}
        for i in range(len(self)):
            graph = self[i]
            for key in graph.node_attrs():
                if graph[key].dtype != torch.float32:
                    continue
                if key not in graph_means:
                    graph_means[key] = []
                    graph_stds[key] = []
                graph_means[key].append(graph[key].mean(dim=0).tolist())
                graph_stds[key].append(graph[key].std(dim=0).tolist())
        means = {
            key: (np.sum(graph_means[key], axis=0) / len(graph_means[key])).tolist()
            for key in graph_means
        }
        stds = {
            key: (
                np.sqrt(
                    np.sum(np.square(graph_stds[key]), axis=0) / len(graph_stds[key])
                )
            ).tolist()
            for key in graph_stds
        }
        return means, stds

    def compute_edge_stats(self) -> tuple[dict[str, float], dict[str, float]]:
        """
        Compute the mean and standard deviation of the edge features.
        """
        graph_means: dict[str, list[float]] = {}
        graph_stds: dict[str, list[float]] = {}
        for i in range(len(self)):
            graph = self[i]
            for key in graph.edge_attrs():
                if graph[key].dtype != torch.float32:
                    continue
                if key not in graph_means:
                    graph_means[key] = []
                    graph_stds[key] = []
                graph_means[key].append(graph[key].mean(dim=0).tolist())
                graph_stds[key].append(graph[key].std(dim=0).tolist())
        means = {
            key: (np.sum(graph_means[key], axis=0) / len(graph_means[key])).tolist()
            for key in graph_means
        }
        stds = {
            key: (
                np.sqrt(
                    np.sum(np.square(graph_stds[key]), axis=0) / len(graph_stds[key])
                )
            ).tolist()
            for key in graph_stds
        }
        return means, stds

    def normalize_inplace(self, graph: PyGGraph) -> None:
        """
        Normalize the features of the graph in place.

        Parameters:
        -----------
        graph: PyGGraph
            The graph to normalize.
        """
        device = graph.pos.device
        for key in graph.node_attrs():
            if key not in self.node_means or key not in self.node_stds:
                continue
            if graph[key].dtype != torch.float32:
                continue
            graph[key] = (
                graph[key]
                - torch.tensor(self.node_means[key], device=device)
            ) / torch.tensor(self.node_stds[key], device=device)
        for key in graph.edge_attrs():
            if key not in self.edge_means or key not in self.edge_stds:
                continue
            if graph[key].dtype != torch.float32:
                continue
            graph[key] = (
                graph[key]
                - torch.tensor(self.edge_means[key], device=device)
            ) / torch.tensor(self.edge_stds[key], device=device)

    def normalize(self, graph: PyGGraph) -> PyGGraph:
        """
        Normalize the features of the graph.

        Parameters:
        -----------
        graph: PyGGraph
            The graph to normalize.

        Returns:
        --------
        PyGGraph: The normalized graph.
        """
        graph = graph.clone()
        self.normalize_inplace(graph)
        return graph

    def denormalize_inplace(self, graph: PyGGraph) -> None:
        """
        Denormalize the features of the graph in place.

        Parameters:
        -----------
        graph: PyGGraph
            The graph to denormalize.
        """

        for key in graph.node_attrs():
            if key not in self.node_means or key not in self.node_stds:
                continue
            if graph[key].dtype != torch.float32:
                continue
            graph[key] = (
                graph[key]
                * torch.tensor(self.node_stds[key], device=graph.device)
            ) + torch.tensor(self.node_means[key], device=graph.device)
        for key in graph.edge_attrs():
            if key not in self.edge_means or key not in self.edge_stds:
                continue
            if graph[key].dtype != torch.float32:
                continue
            graph[key] = (
                graph[key]
                * torch.tensor(self.edge_stds[key], device=graph.device)
            ) + torch.tensor(self.edge_means[key], device=graph.device)

    def denormalize(self, graph: PyGGraph) -> PyGGraph:
        """
        Denormalize the features of the graph.

        Parameters:
        -----------
        graph: PyGGraph
            The graph to denormalize.

        Returns:
        --------
        PyGGraph: The denormalized graph.
        """
        graph = graph.clone()
        self.denormalize_inplace(graph)
        return graph

    def PyG_to_pyvista_polydata(self, graph: PyGGraph) -> pv.PolyData:
        """
        Convert a PyGGraph of the volume flow field to a Pyvista PolyData object. This will not be the original mesh since the graph only contains the cell centers and the connectivity, not the cell points

        Parameters:
        -----------
        graph: PyGGraph
            The PyGGraph to convert.

        Returns:
        --------
        pv.PolyData: The converted PolyData object.
        """
        if self.do_normalization:
            graph = self.denormalize(graph)
        graph = graph.to("cpu")
        if graph.edge_index is None or graph.edge_index.size(0) != 2:
            raise ValueError(
                "Graph must have an edge_index attribute with shape [2, num_edges]"
            )
        e_from, e_to = graph.edge_index[0], graph.edge_index[1]
        num_nodes_per_edge = torch.ones_like(e_from) * 2
        edges = torch.stack([num_nodes_per_edge, e_from, e_to], dim=1).detach()
        grid = pv.PolyData(graph.pos.detach().numpy(), edges.numpy())
        grid.point_data["Pressure"] = graph["Pressure"].detach().numpy()
        grid.point_data["WallShearStress_0"] = (
            graph["ShearStress"][:, 0].detach().numpy()
        )
        grid.point_data["WallShearStress_1"] = (
            graph["ShearStress"][:, 1].detach().numpy()
        )
        grid.point_data["WallShearStress_2"] = (
            graph["ShearStress"][:, 2].detach().numpy()
        )
        grid.point_data["Temperature"] = graph["Temperature"].detach().numpy()
        grid.point_data["Normal_0"] = graph["Normal"][:, 0].detach().numpy()
        grid.point_data["Normal_1"] = graph["Normal"][:, 1].detach().numpy()
        grid.point_data["Normal_2"] = graph["Normal"][:, 2].detach().numpy()
        grid.point_data["CellArea"] = graph["CellArea"].detach().numpy()
        grid.point_data["BodyID"] = graph["BodyID"].detach().numpy()
        return grid

    def plot_surface(
        self,
        graph: PyGGraph,
        scalar: SurfaceFieldType,
        object_id: Optional[int] = None,
    ):
        """
        Plot the surface of the graph.

        Parameters:
        -----------
        graph: PyGGraph
            The graph to plot.
        """
        if object_id is not None:
            mask = graph["BodyID"] == object_id
            graph = graph.subgraph(mask)
        grid = self.PyG_to_pyvista_polydata(graph)
        grid.plot(
            scalars=scalar,
            render_points_as_spheres=True,
            eye_dome_lighting=True,
            render_lines_as_tubes=True,
            line_width=4,
            style="wireframe",
        )

    def compute_aggregate_force(
        self, graph: PyGGraph, object_id: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compute the aggregate force acting on the surface.

        Parameters:
        -----------
        graph: PyGGraph
            The graph to compute the aggregate force for.
        object_id: Optional[int]
            The object id to compute the aggregate force for. If None, the aggregate force for all objects is computed.

        Returns:
        --------
        torch.Tensor: The aggregate force acting on the surface.
        """
        if self.do_normalization:
            graph = self.denormalize(graph)
        cell_areas = graph["CellArea"]

        cell_pressure_forces = (
            graph["Pressure"][:, None]
            * cell_areas[:, None]
            * graph["Normal"]
        )
        cell_shear_forces = graph["ShearStress"] * cell_areas[:, None]
        if object_id is not None:
            mask = graph["BodyID"] == object_id
            cell_pressure_forces = cell_pressure_forces[mask]
            cell_shear_forces = cell_shear_forces[mask]
        pressure_force = cell_pressure_forces.sum(dim=0)
        shear_force = cell_shear_forces.sum(dim=0)
        return pressure_force + shear_force

    @classmethod
    def l2_loss(cls, graph1: PyGGraph, graph2: PyGGraph):
        """
        Compute the L2 loss between two PyGGraphs.

        Parameters:
        -----------
        graph1: PyGGraph
            The first graph.
        graph2: PyGGraph
            The second graph.

        Returns:
        --------
        torch.tensor: The L2 loss between the two graphs.
        """
        # check that the graphs have the same number of nodes
        assert graph1.num_nodes == graph2.num_nodes
        # check that the graphs have the same number of edges
        assert graph1.num_edges == graph2.num_edges

        return (
            torch.nn.functional.mse_loss(
                graph1["Pressure"], graph2["Pressure"]
            )
            + torch.nn.functional.mse_loss(
                graph1["Temperature"], graph2["Temperature"]
            )
            + torch.nn.functional.mse_loss(
                graph1["ShearStress"], graph2["ShearStress"]
            )
        )
