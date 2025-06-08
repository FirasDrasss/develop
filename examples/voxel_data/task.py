import lightning.pytorch as pl
import torch


class VoxelizedFlowFieldPredictionTask(pl.LightningModule):
    def __init__(self, model, learning_rate=0.001):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        mask, Y = batch
        mask = mask.unsqueeze(1).float().to(self.device)
        Y = Y.permute(0, 4, 1, 2, 3).to(self.device)
        pred = self.model(mask.float())
        loss = torch.nn.functional.mse_loss(pred, Y)
        self.log("train_loss", loss, batch_size=mask.shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        mask, Y = batch
        mask = mask.unsqueeze(1).float().to(self.device)
        Y = Y.permute(0, 4, 1, 2, 3).to(self.device)
        pred = self.model(mask.float())
        loss = torch.nn.functional.mse_loss(pred, Y)
        self.log("val_loss", loss, batch_size=mask.shape[0])
        return loss

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )
        return self.optimizer
