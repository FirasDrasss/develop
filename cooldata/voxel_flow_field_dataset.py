import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, Union

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import torch
import torch.utils
import torch.utils.data
from ipywidgets import fixed, interact
from matplotlib import cm
from tensordict import TensorDict
from tqdm import tqdm

from cooldata.pyvista_flow_field_dataset import PyvistaFlowFieldDataset, PyvistaSample

VoxelField = Literal[
    "Pressure",
    "Temperature",
    "Velocity",
    "Position",
    "TurbulentKineticEnergy",
    "TurbulentDissipationRate",
]
voxel_fields: list[VoxelField] = [
    "Pressure",
    "Temperature",
    "Velocity",
    "Position",
    "TurbulentKineticEnergy",
    "TurbulentDissipationRate",
]

Normalization = dict[
    VoxelField, Union[tuple[list[float], list[float]], tuple[float, float]]
]
"""
A dictionary that maps field names to tuples of (mean, std) for normalization.
"""


class VoxelFlowFieldSample:
    def __init__(
        self,
        path: str,
        bounding_box: tuple[float, float, float, float, float, float],
        resolution: tuple[int, int, int],
        normalization: Normalization | None = None,
    ):
        """A sample of a voxelized flow field. The data is stored in a file in the TensorDict format on disk in an unnormalized form. A normalization can be applied dynamically when getting data.

        Args:
        - path: Path to the file containing the voxelized flow field.
        - bounding_box: Tuple (xmin, xmax, ymin, ymax, zmin, zmax) defining the bounding box of the flow field.
        - resolution: Tuple (nx, ny, nz) defining the resolution of the voxel grid.
        - normalization: A normalization that is applied dynamically when getting data. A dictionary that maps field names to tuples of (mean, std) for normalization. The
        """

        self.path = path
        self.bounding_box = bounding_box
        self.resolution = resolution
        self._data: TensorDict | None = None
        self.normalization = normalization

    @property
    def data(self):
        if self._data is None:
            self._data = TensorDict.load(self.path)
            # TODO: Check if data is a valid flow field dataset, i.e., has the necessary point data
        return self._data

    def get_field(self, field: VoxelField, normalized=True) -> torch.Tensor:
        """
        Returns the specified field of the flow field as a torch.Tensor.
        Shape:
        - Pressure, Temperature: (resolution_x, resolution_y, resolution_z)
        - Velocity, Position: (resolution_x, resolution_y, resolution_z, 3)
        """
        if self.normalization is not None and normalized:
            mean, std = self.normalization[field]
            return (
                self.data[field] - torch.tensor(mean, device=self.data.device)
            ) / torch.tensor(std, device=self.data.device)
        return self.data[field]

    @property
    def Y(self):
        return torch.cat(
            [
                self.get_field("Velocity"),
                self.get_field("Pressure").unsqueeze(-1),
                self.get_field("Temperature").unsqueeze(-1),
            ],
            dim=-1,
        )

    @property
    def mask(self):
        return self.data["Mask"]

    @classmethod
    def from_mask_y(
        cls,
        mask: torch.Tensor,
        Y: torch.Tensor,
        bounding_box: tuple[float, float, float, float, float, float],
        resolution: tuple[int, int, int],
        save_path: str,
        normalization: Normalization | None = None,
    ) -> "VoxelFlowFieldSample":
        assert mask.shape == tuple(resolution), (
            f"Mask shape {mask.shape} does not match resolution {resolution}"
        )
        assert Y.shape == (resolution[0], resolution[1], resolution[2], 5), (
            f"Y shape {Y.shape} does not match resolution {resolution}"
        )
        # check if the mask is a boolean tensor
        assert mask.dtype == torch.bool, f"Mask dtype {mask.dtype} is not boolean"
        assert Y.dtype == torch.float32, f"Y dtype {Y.dtype} is not float32"
        velocity = Y[:, :, :, :3]
        pressure = Y[:, :, :, 3]
        temperature = Y[:, :, :, 4]
        # get position from bounding box and resolution
        xmin, xmax, ymin, ymax, zmin, zmax = bounding_box
        x = np.linspace(xmin, xmax, resolution[0])
        y = np.linspace(ymin, ymax, resolution[1])
        z = np.linspace(zmin, zmax, resolution[2])
        x, y, z = np.meshgrid(x, y, z, indexing="ij")
        position = torch.tensor(np.stack([x, y, z], axis=-1), dtype=torch.float32)
        data = TensorDict(
            {
                "Pressure": pressure,
                "Temperature": temperature,
                "Velocity": velocity,
                "Mask": mask,
                "Position": position,
            }
        )
        data = data.to("cpu")
        # denormalize the data
        if normalization is not None:
            for field in data.keys():
                if field in normalization:
                    mean, std = normalization[field]
                    data[field] = data[field] * torch.tensor(
                        std, device=data.device
                    ) + torch.tensor(mean, device=data.device)
        data.save(save_path)
        return cls(save_path, bounding_box, resolution, normalization)

    @classmethod
    def from_pyvista(
        cls,
        sample: PyvistaSample,
        save_path: str,
        resolution: tuple[int, int, int],
        bounding_box: tuple[float, float, float, float, float, float],
    ) -> "VoxelFlowFieldSample":
        """
        Interpolates the volume data from the sample to a voxel grid and saves it to a file.
        """
        if os.path.exists(save_path):
            return cls(save_path, bounding_box, resolution)
        xmin, xmax, ymin, ymax, zmin, zmax = bounding_box
        x = np.linspace(xmin, xmax, resolution[0])
        y = np.linspace(ymin, ymax, resolution[1])
        z = np.linspace(zmin, zmax, resolution[2])
        x, y, z = np.meshgrid(x, y, z, indexing="ij")
        grid = pv.StructuredGrid(x, y, z)
        was_loaded = sample.is_loaded
        volume_data = sample.volume_data[0][0][0]
        interpolated = grid.sample(volume_data)
        velocities = [interpolated[f"Velocity_{i}"] for i in range(3)]
        data = TensorDict(
            {
                "Pressure": torch.tensor(
                    interpolated["Pressure"].reshape(resolution, order="F"),
                    dtype=torch.float32,
                ),
                "Temperature": torch.tensor(
                    interpolated["Temperature"].reshape(resolution, order="F"),
                    dtype=torch.float32,
                ),
                "Velocity": torch.stack(
                    [
                        torch.tensor(
                            v.reshape(resolution, order="F"), dtype=torch.float32
                        )
                        for v in velocities
                    ],
                    dim=-1,
                ),
                "Mask": torch.tensor(
                    interpolated["vtkValidPointMask"].reshape(resolution, order="F"),
                    dtype=torch.bool,
                ),
                "Position": torch.tensor(
                    np.stack([x, y, z], axis=-1).reshape(resolution + (3,), order="F"),
                    dtype=torch.float32,
                ),
                "TurbulentKineticEnergy": torch.tensor(
                    interpolated["TurbulentKineticEnergy"].reshape(
                        resolution, order="F"
                    ),
                    dtype=torch.float32,
                ),
                "TurbulentDissipationRate": torch.tensor(
                    interpolated["TurbulentDissipationRate"].reshape(
                        resolution, order="F"
                    ),
                    dtype=torch.float32,
                ),
            }
        )
        data.save(save_path)
        if not was_loaded:
            sample.unload()
        return cls(save_path, bounding_box, resolution)

    def load(self):
        self._data
        return self

    def unload(self):
        self._data = None
        return self

    def to_pyvista(self):
        xmin, xmax, ymin, ymax, zmin, zmax = self.bounding_box
        x, y, z = np.mgrid[
            xmin : xmax : complex(self.resolution[0]),
            ymin : ymax : complex(self.resolution[1]),
            zmin : zmax : complex(self.resolution[2]),
        ]
        grid = pv.StructuredGrid(x, y, z)
        grid["Pressure"] = self.data["Pressure"].numpy().flatten(order="F")
        grid["Temperature"] = self.data["Temperature"].numpy().flatten(order="F")
        grid["Velocity"] = self.data["Velocity"].numpy().reshape(-1, 3, order="F")
        grid["vtkValidPointMask"] = (
            self.data["Mask"].numpy().flatten(order="F").astype(np.int8)
        )
        grid["vtkGhostType"] = np.zeros(len(grid.points), dtype=np.uint8)
        grid["vtkGhostType"][~self.data["Mask"].numpy().flatten(order="F")] = 32
        grid.cell_data["vtkGhostType"] = np.zeros(grid.n_cells, dtype=np.uint8)
        grid.cell_data["vtkGhostType"][
            ~self.data["Mask"].numpy()[1:, 1:, 1:].flatten(order="F")
        ] = 32
        return grid

    def plot(self, field: VoxelField):
        grid = self.to_pyvista()
        grid.plot(scalars=field, cmap="viridis")

    def plot_slice(
        self,
        field: VoxelField,
        slice_idx: int | None = None,
        axis: Literal["x", "y", "z"] = "z",
    ):
        """
        Plots a slice of the field at the specified index along the specified axis.
        Args:
        - field: The field to plot.
        - slice_idx: The index of the slice to plot. If None, the middle slice is used.
        - axis: The axis along which to plot the slice. Can be "x", "y", or "z".
        """
        field_values = self.get_field(field, normalized=False)
        field_np = field_values.cpu().numpy()
        slice: np.ndarray | None = None
        if axis == "x":
            if slice_idx is None:
                slice_idx = field_np.shape[0] // 2
            slice = field_np[slice_idx, :, :]
        elif axis == "y":
            if slice_idx is None:
                slice_idx = field_np.shape[1] // 2
            slice = field_np[:, slice_idx, :]
        elif axis == "z":
            if slice_idx is None:
                slice_idx = field_np.shape[2] // 2
            slice = field_np[:, :, slice_idx]
        else:
            raise ValueError("Axis must be 'x', 'y', or 'z'.")
        title_addon = ""
        if slice.ndim == 3:
            slice = np.sqrt(
                np.sum(slice**2, axis=-1)
            )  # If it's a vector field, take the magnitude
            title_addon = " (magnitude)"
        plt.imshow(slice, cmap="viridis")
        plt.colorbar()
        if axis == "x":
            plt.xlabel("y")
            plt.ylabel("z")
            plt.yticks(
                ticks=np.arange(0, self.resolution[1], step=5),
                labels=[
                    f"{val:.2f}"
                    for val in np.linspace(
                        self.bounding_box[2],
                        self.bounding_box[3],
                        num=self.resolution[1],
                    )[::5]
                ],
            )
            plt.xticks(
                ticks=np.arange(0, self.resolution[2], step=5),
                labels=[
                    f"{val:.2f}"
                    for val in np.linspace(
                        self.bounding_box[4],
                        self.bounding_box[5],
                        num=self.resolution[2],
                    )[::5]
                ],
            )
            plt.title(
                f"{field} at x = {self.bounding_box[0] + (slice_idx / self.resolution[0]) * (self.bounding_box[1] - self.bounding_box[0]):.2f}{title_addon}"
            )
        elif axis == "y":
            plt.xlabel("x")
            plt.ylabel("z")
            plt.yticks(
                ticks=np.arange(0, self.resolution[0], step=5),
                labels=[
                    f"{val:.2f}"
                    for val in np.linspace(
                        self.bounding_box[0],
                        self.bounding_box[1],
                        num=self.resolution[0],
                    )[::5]
                ],
            )
            plt.xticks(
                ticks=np.arange(0, self.resolution[2], step=5),
                labels=[
                    f"{val:.2f}"
                    for val in np.linspace(
                        self.bounding_box[4],
                        self.bounding_box[5],
                        num=self.resolution[2],
                    )[::5]
                ],
            )
            plt.title(
                f"{field} at y = {self.bounding_box[2] + (slice_idx / self.resolution[1]) * (self.bounding_box[3] - self.bounding_box[2]):.2f}{title_addon}"
            )
        elif axis == "z":
            plt.xlabel("x")
            plt.ylabel("y")
            plt.yticks(
                ticks=np.arange(0, self.resolution[0], step=5),
                labels=[
                    f"{val:.2f}"
                    for val in np.linspace(
                        self.bounding_box[0],
                        self.bounding_box[1],
                        num=self.resolution[0],
                    )[::5]
                ],
            )
            plt.xticks(
                ticks=np.arange(0, self.resolution[1], step=5),
                labels=[
                    f"{val:.2f}"
                    for val in np.linspace(
                        self.bounding_box[2],
                        self.bounding_box[3],
                        num=self.resolution[1],
                    )[::5]
                ],
            )
            plt.title(
                f"{field} at z = {self.bounding_box[4] + (slice_idx / self.resolution[2]) * (self.bounding_box[5] - self.bounding_box[4]):.2f}{title_addon}"
            )

        # preserve aspect ratio
        plt.gca().set_aspect("equal", adjustable="box")
        plt.grid()
        plt.tight_layout()
        plt.show()

    def plot_slice_interactively(
        self, field: VoxelField, axis: Literal["x", "y", "z"] = "z"
    ):
        """
        Plots a slice of the field at the specified index along the specified axis.
        Args:
        - field: The field to plot.
        - axis: The axis along which to plot the slice. Can be "x", "y", or "z".
        """

        def update_plot(slice_idx):
            self.plot_slice(field, slice_idx, axis)

        interact(
            update_plot,
            slice_idx=widgets.IntSlider(
                min=0,
                max=self.resolution[{"x": 0, "y": 1, "z": 2}[axis]] - 1,
                step=1,
                value=self.resolution[{"x": 0, "y": 1, "z": 2}[axis]] // 2,
            ),
            field=fixed(field),
            axis=fixed(axis),
        )


@dataclass
class VoxelFlowFieldDatasetConfig:
    """Configuration for creating a VoxelFlowFieldDataset from a PyvistaFlowFieldDataset."""

    pyvista_dataset: PyvistaFlowFieldDataset
    resolution: tuple[int, int, int] = (32, 32, 32)


class VoxelFlowFieldDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        cache_dir: str,
        config: VoxelFlowFieldDatasetConfig | None = None,
        resume_loading_from_cache: bool = True,
    ):
        """
        Dataset of voxelized flow fields. The constructor either loads the dataset from a cache directory or converts a
        PyvistaFlowFieldDataset to a DGLFlowFieldDataset.
        """
        self.cache_dir = Path(os.path.abspath(cache_dir))
        if not self.cache_dir.exists():
            self.cache_dir.mkdir(parents=True)
        self.samples: list[VoxelFlowFieldSample] = []
        if config is not None:
            # clear the cache directory
            if not resume_loading_from_cache and self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            config.pyvista_dataset.unload()
            self.resolution = config.resolution
            if os.path.exists(os.path.join(self.cache_dir, "metadata.json")):
                metadata = json.load(
                    open(os.path.join(self.cache_dir, "metadata.json"))
                )
                if tuple(metadata["resolution"]) == config.resolution:
                    # If the metadata matches, we can use it
                    self.resolution = metadata["resolution"]
                    self.bounding_box = metadata["bounding_box"]
                    self.normalization = metadata["normalization"]
            if not hasattr(self, "bounding_box"):
                # If the bounding box is not set, we use the bounding box of the PyvistaFlowFieldDataset
                self.bounding_box = config.pyvista_dataset.get_bounds()

            args_list = [
                (
                    config.pyvista_dataset[i],
                    self.cache_dir,
                    i,
                    config.resolution,
                    self.bounding_box,
                )
                for i in range(len(config.pyvista_dataset))
            ]
            with ProcessPoolExecutor() as executor:
                results = list(
                    tqdm(
                        executor.map(_create_voxel_sample, args_list),
                        total=len(args_list),
                        desc="Voxelizing samples",
                    )
                )
            self.samples.extend(results)
            if not hasattr(self, "normalization"):
                self.normalization = self.compute_normalization()

            # else, we compute the normalization and save the metadata
            json.dump(
                {
                    "resolution": config.resolution,
                    "bounding_box": self.bounding_box,
                    "normalization": self.normalization,
                },
                open(os.path.join(self.cache_dir, "metadata.json"), "w"),
            )
        else:
            metadata = json.load(open(os.path.join(self.cache_dir, "metadata.json")))
            self.resolution = metadata["resolution"]
            self.bounding_box = metadata["bounding_box"]
            self.normalization = metadata["normalization"]
            for file in os.listdir(self.cache_dir):
                if file.endswith(".pt"):
                    self.samples.append(
                        VoxelFlowFieldSample(
                            os.path.join(self.cache_dir, file),
                            self.bounding_box,
                            self.resolution,
                        )
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int | slice | list[int]):
        if isinstance(idx, slice):
            return self.slice(idx.start, idx.stop)
        if isinstance(idx, list):
            new_ds = VoxelFlowFieldDataset(self.cache_dir)
            new_ds.samples = [self.samples[i] for i in idx]
            new_ds.bounding_box = self.bounding_box
            new_ds.resolution = self.resolution
            new_ds.normalization = self.normalization
            return new_ds
        if idx < 0 or idx >= len(self.samples):
            raise IndexError(
                f"Index {idx} out of bounds for dataset of length {len(self.samples)}"
            )
        if not isinstance(idx, int):
            raise TypeError(f"Index must be an integer or a slice, got {type(idx)}")
        return self.samples[idx]

    def slice(self, start: int, end: int):
        """
        Returns a slice of the dataset.

        Args:
        - start: The start index of the slice.
        - end: The end index of the slice.

        Returns:
        A new VoxelFlowFieldDataset containing the specified slice.
        """
        new_ds = VoxelFlowFieldDataset(self.cache_dir)
        new_ds.samples = self.samples[start:end]
        new_ds.bounding_box = self.bounding_box
        new_ds.resolution = self.resolution
        new_ds.normalization = self.normalization
        return new_ds

    def shuffle(self):
        """Shuffles the dataset in place."""
        np.random.shuffle(self.samples)
        return self

    def compute_normalization(self) -> Normalization:
        normalization: Normalization = {}
        for field in voxel_fields:
            sample_means = []
            sample_stds = []
            for sample in self.samples:
                sample_means.append(sample.get_field(field).mean(dim=(0, 1, 2)))
                sample_stds.append(sample.get_field(field).std(dim=(0, 1, 2)))
            mean = torch.stack(sample_means).mean(dim=0).tolist()
            std = torch.stack(sample_stds).mean(dim=0).tolist()
            normalization[field] = (mean, std)
        return normalization

    def normalize(self):
        """Normalizes the dataset in place."""
        for sample in self.samples:
            sample.normalization = self.normalization
        return self

    def prediction_to_sample(self, mask: torch.Tensor, Y: torch.Tensor):
        """
        Converts a prediction to a sample. The mask is used to create a new sample with the same bounding box and resolution as the original dataset.

        Args:
        - mask: The mask of the prediction. Shape: (resolution_x, resolution_y, resolution_z)
        - Y: The prediction. Shape: (resolution_x, resolution_y, resolution_z, 5)
        """
        # get the bounding box and resolution from the original dataset
        bounding_box = self.bounding_box
        resolution = self.resolution
        now = datetime.now()

        # create a new sample from the prediction
        timestamp_formatted = now.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        return VoxelFlowFieldSample.from_mask_y(
            mask,
            Y,
            bounding_box,
            resolution,
            os.path.join(
                self.cache_dir, "predictions", f"prediction_{timestamp_formatted}.pt"
            ),
            normalization=self.normalization,
        )

    def unnormalize(self):
        """Unnormalizes the dataset in place."""
        for sample in self.samples:
            sample.normalization = None
        return self

    def get_default_loadable_dataset(self):
        """Get a dataset that returns the mask as X and all flow features concatenated as Y"""
        return DefaultVoxelDataset(self)


class DefaultVoxelDataset(torch.utils.data.Dataset):
    def __init__(self, ds: VoxelFlowFieldDataset):
        super().__init__()
        self.ds = ds

    def __getitem__(self, index: int):
        item = self.ds[index]
        return item.mask, item.Y

    def __len__(self):
        return len(self.ds)


def _create_voxel_sample(args):
    sample_pv, cache_dir, i, resolution, bounding_box = args
    return VoxelFlowFieldSample.from_pyvista(
        sample_pv,
        os.path.join(cache_dir, f"{i}.pt"),
        resolution,
        bounding_box,
    )
