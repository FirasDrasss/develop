from pathlib import Path
from typing import Literal
import dgl
# from dgl.data import DGLDataset

import torch
from torch.utils.data import Dataset
# from jaxtyping import Float32, Array

import h5py
import torch

class FlowGeometry:
    """
    Represents the geometry of a flow field, including mesh structure and boundary data.

    This class contains point coordinates, edge connectivity, and detailed boundary condition
    information. It also stores surface-related data like normals and surface associations.

    Attributes:
        points (torch.Tensor): Tensor of shape (num_points, 3) with 3D coordinates of each point.
        connectivity (torch.Tensor): Tensor of shape (num_edges, 2), dtype=torch.int32.
            Each row defines an edge between two points by their indices.
        boundary_conditions (torch.Tensor): Tensor of shape (num_edges,), dtype=torch.int8.
            Indicates boundary type per edge:
                - 0: Internal
                - 1: Inflow
                - 2: Outflow
                - 3+: Wall (body index)
        surface_indices (torch.Tensor): Tensor of shape (num_surface_points,), dtype=torch.int32.
            Associates each surface point with a surface/body.
        surface_normals (torch.Tensor): Tensor of shape (num_surface_points, 3), dtype=torch.float32.
            Contains normal vectors at each surface point.
    """
    def __init__(
        self,
        points: torch.Tensor,
        connectivity: torch.Tensor,
        boundary_conditions: torch.Tensor,
        surface_indices: torch.Tensor,
        surface_normals: torch.Tensor,
    ) -> None:
        """
        Initializes the FlowGeometry object with point cloud and connectivity data.

        Args:
            points (torch.Tensor): Tensor of shape (num_points, 3), dtype=torch.float32.
                3D coordinates of all points.
            connectivity (torch.Tensor): Tensor of shape (num_edges, 2), dtype=torch.int32.
                Edge definitions using point indices.
            boundary_conditions (torch.Tensor): Tensor of shape (num_edges,), dtype=torch.int8.
                Type of each edge (0 = internal, 1 = inflow, 2 = outflow, 3+ = wall).
            surface_indices (torch.Tensor): Tensor of shape (num_surface_points,), dtype=torch.int32.
                Index of the surface/body each surface point belongs to.
            surface_normals (torch.Tensor): Tensor of shape (num_surface_points, 3), dtype=torch.float32.
                Normal vector of the surface at each surface point.
        """
        num_points = points.shape[0]
        assert points.shape == (num_points, 3)
        assert points.dtype == torch.float32
        num_edges = connectivity.shape[0]
        assert connectivity.shape == (num_edges, 2)
        assert connectivity.dtype == torch.int32
        assert boundary_conditions.shape == (num_edges,)
        assert boundary_conditions.dtype == torch.int8
        assert torch.all(boundary_conditions >= 0)
        assert torch.all(boundary_conditions <= 3)
        self.points = points
        self.connectivity = connectivity
        self.boundary_conditions = boundary_conditions
        
        num_surface_points = surface_indices.shape[0]
        assert surface_indices.shape == (num_surface_points,)
        assert surface_indices.dtype == torch.int32
        assert surface_normals.shape == (num_surface_points, 3)
        assert surface_normals.dtype == torch.float32
        self.surface_indices = surface_indices
        self.surface_normals = surface_normals
    @property
    def num_points(self):
        """
        Returns the number of points in the geometry.

        Returns:
            int: Total number of points.
        """
        return self.points.shape[0]
    @property
    def num_edges(self):
        """
        Returns the number of edges in the geometry.

        Returns:
            int: Total number of edges.
        """
        return self.connectivity.shape[0]
    

class FlowField:
    def __init__(
        self,
        geometry: FlowGeometry,
        velocities: torch.Tensor,
        pressures: torch.Tensor,
        temperatues: torch.Tensor,
    ) -> None:
        """
        A single flow field.
        
        Parameters:
        - geometry: FlowGeometry
        - velocities: torch.Tensor of shape (num_points, 3)
        - pressures: torch.Tensor of shape (num_points,)
        - temperatues: torch.Tensor of shape (num_points,)
        """
        self.geometry = geometry
        num_points = self.geometry.num_points
        assert velocities.shape == (num_points, 3)
        assert velocities.dtype == torch.float32
        assert pressures.shape == (num_points,)
        assert pressures.dtype == torch.float32
        assert temperatues.shape == (num_points,)
        assert temperatues.dtype == torch.float32
        self.velocities = velocities
        self.pressures = pressures
        self.temperatues = temperatues
        

    def compute_pressure_force(self, body_id: int):
        """
        Compute the pressure force vector acting on the surface of the body with the given id.
        Should be differentiable.
        """
        # Compute the pressure gradient
        # Compute the pressure force
        raise NotImplementedError()
        


class FlowFieldDataset(Dataset):
    def __init__(
        self,
        name: str,
        variant: Literal["full", "geometry1"],
        resolution: Literal["full", "half"],
        url=None,
        raw_dir=None,
        save_dir=None,
        hash_key=(),
        force_reload=False,
        verbose=False,
        transform=None,
    ) -> None:
        """
        Load a flow field dataset from a given path to a .hdf5 file.
        """
        super().__init__(name, url, raw_dir, save_dir, hash_key, force_reload, verbose)

    def download(self):
        assert self.url is not None, "No URL provided for dataset download"
        # TODO: download the dataset from the given url to the raw_dir
        # self.url should point to a .hdf5 file containing the dataset

        raise NotImplementedError()

    def process(self):
        assert self.raw_dir is not None, "No raw_dir provided for dataset processing"
