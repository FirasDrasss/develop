from pathlib import Path

import lightning.pytorch as pl
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import DataLoader as Dataloader

from cooldata.pyvista_flow_field_dataset import PyvistaFlowFieldDataset
from cooldata.voxel_flow_field_dataset import (
    VoxelFlowFieldDataset,
    VoxelFlowFieldDatasetConfig,
)


class VoxelDataModule(pl.LightningDataModule):
    def __init__(self, save_dir: str, num_samples: int, batch_size: int) -> None:
        super().__init__()
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self) -> None:
        ds_pv = PyvistaFlowFieldDataset.load_from_huggingface(
            num_samples=self.num_samples,
            data_dir=self.save_dir / f"pyvista-{self.num_samples}",
        )
        ds_voxel = VoxelFlowFieldDataset(
            cache_dir=str(self.save_dir / f"voxels-{self.num_samples}"),
            config=VoxelFlowFieldDatasetConfig(
                pyvista_dataset=ds_pv, resolution=(32, 16, 16)
            ),
        )
        ds_voxel.normalize()
        num_train_samples = int(len(ds_voxel) * 0.8)
        num_val_samples = int((len(ds_voxel) - num_train_samples) * 0.5)
        num_test_samples = len(ds_voxel) - num_train_samples - num_val_samples
        ds_voxel.shuffle()
        self.train_dataset = ds_voxel[:num_train_samples]
        self.val_dataset = ds_voxel[
            num_train_samples : num_train_samples + num_val_samples
        ]
        self.test_dataset = ds_voxel[num_train_samples + num_val_samples :]

    def train_dataloader(self):
        return Dataloader(
            self.train_dataset.get_default_loadable_dataset(),
            batch_size=self.batch_size,
        )

    def val_dataloader(self):
        return Dataloader(
            self.val_dataset.get_default_loadable_dataset(), batch_size=self.batch_size
        )

    def test_dataloader(self):
        return Dataloader(
            self.test_dataset.get_default_loadable_dataset(), batch_size=self.batch_size
        )
