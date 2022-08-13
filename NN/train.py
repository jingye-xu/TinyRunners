
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch import optim, nn, utils, Tensor
import pytorch_lightning as pl
from model.model_1 import IEGMNet
from model.dataset_load import DatasetTiny


class TinyModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = IEGMNet()


    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self.net(x)
        loss = F.cross_entropy(outputs, y)
        
        # # Logging to TensorBoard by default
        self.log("train_loss", loss)
        
        return loss
        
    def forward(self, x):
        return self.net(x)
        
    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y = batch
        outputs = self.net(x)
        loss = F.cross_entropy(outputs, y)
        
        self.log("val_loss", loss)
        
        
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer



dataset = DatasetTiny("data")
print(len(dataset))

val_count = 300
batch_size = 128
data_train, data_val = random_split(dataset, [len(dataset)-val_count, val_count])

train_loader = DataLoader(data_train, batch_size=batch_size)
val_loader = DataLoader(data_val, batch_size=batch_size)

# model
model = TinyModel()

# training
training_device = "gpu" if torch.cuda.is_available() else "cpu"
trainer = pl.Trainer(
    accelerator=training_device, 
    devices=1, 
    num_nodes=1, 
    precision=16, 
    limit_train_batches=0.5,
    max_epochs = 20
    )
trainer.fit(model, train_loader, val_loader)

# export model
input_sample = torch.rand((1, 1,1250,1))
model.to_onnx("lightning_logs/model.onnx", input_sample, export_params=True)
