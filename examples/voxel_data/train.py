import os
import sys

base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.append(base_path)
import numpy as np
import pandas as pd
import torch
from flow_field_model import create_flow_field_model
from model import FlowFieldTransformer, count_parameters
from torch.utils.data import DataLoader

from cooldata.pyvista_flow_field_dataset import PyvistaFlowFieldDataset
from cooldata.voxel_flow_field_dataset import (
    VoxelFlowFieldDataset,
    VoxelFlowFieldDatasetConfig,
)

if __name__ == "__main__":
    ds_pv = PyvistaFlowFieldDataset.load_from_huggingface(
        num_samples=100, data_dir="datasets/pyvista-medium"
    )
    ds_voxel = VoxelFlowFieldDataset(
        cache_dir="datasets/voxels-medium",
        config=VoxelFlowFieldDatasetConfig(
            pyvista_dataset=ds_pv, resolution=(32, 16, 16)
        ),
    )
    # ds_voxel = VoxelFlowFieldDataset(cache_dir="datasets/voxels-medium")
    ds_voxel.normalize()
    ds_voxel.shuffle()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    num_train_samples = int(len(ds_voxel) * 0.8)
    num_val_samples = len(ds_voxel) - num_train_samples
    np.random.seed(42)
    ds_voxel.shuffle()
    train_dataset = ds_voxel[:num_train_samples]
    val_dataset = ds_voxel[num_train_samples:]
    train_dataloader = DataLoader(
        train_dataset.get_default_loadable_dataset(), batch_size=16, shuffle=False
    )
    val_dataloader = DataLoader(
        val_dataset.get_default_loadable_dataset(), batch_size=16, shuffle=False
    )

    model = create_flow_field_model(input_shape=(32, 16, 16), out_channels=5)
    num_params = count_parameters(model)
    print(f"Number of parameters: {num_params}")
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    losses = []
    val_losses = []
    for epoch in range(100):
        model.train()
        loss_sum = 0.0
        for batch in train_dataloader:
            mask, Y = batch
            mask = mask.unsqueeze(1).float().to(device)
            Y = Y.permute(0, 4, 1, 2, 3).to(device)
            pred = model(mask.float())
            loss = torch.nn.functional.mse_loss(pred, Y, reduction="mean")
            loss_sum += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        losses.append(loss_sum / len(train_dataset))
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for batch in val_dataloader:
                mask, Y = batch
                mask = mask.unsqueeze(1).float().to(device)
                Y = Y.permute(0, 4, 1, 2, 3).to(device)
                pred = model(mask)
                val_loss += torch.nn.functional.mse_loss(
                    pred, Y, reduction="mean"
                ).item()
            val_loss /= len(val_dataset)
            val_losses.append(val_loss)
        print(f"Epoch {epoch}, Loss: {losses[-1]}, Val Loss: {val_loss}")
    df = pd.DataFrame(
        {
            "train_loss": losses,
            "validation_loss": val_losses,
            "epoch": list(range(len(losses))),
        }
    )
    df.to_csv("figures/losses.csv", index=False)
