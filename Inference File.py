
# AI600 - Assignment 2 - Inference Script
# Roll No: 25280081 
# Champion Model: 784 -> 384 -> 256 -> 128 -> 15


import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# Champion Model Architecture 
class ChampionMLP(nn.Module):
    def __init__(self, input_dim=784, num_classes=15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 384), nn.BatchNorm1d(384), nn.GELU(), nn.Dropout(0.35),
            nn.Linear(384, 256),       nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.25),
            nn.Linear(256, 128),       nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x.view(-1, 784))


# Load Model
model = ChampionMLP()
model.load_state_dict(torch.load("champion_weights.pt", map_location="cpu"))
model.eval()
print("Model loaded successfully (436,367 parameters)")


# Load Test Data
test_npz = np.load("quickdraw_test.npz")
x_test = torch.tensor(test_npz["test_images"].astype(np.float32) / 255.0)
print(f"Test data loaded: {x_test.shape[0]} samples")


# Generate Predictions
predictions = []
with torch.no_grad():
    for (xb,) in DataLoader(TensorDataset(x_test), batch_size=512, shuffle=False):
        preds = model(xb).argmax(dim=1)
        predictions.append(preds)

predictions = torch.cat(predictions).numpy()
print(f"Predictions generated: {len(predictions)} samples")
print(f"Class range: [{predictions.min()}, {predictions.max()}]")


# Save Predictions
np.savetxt("predictions.csv", predictions, fmt="%d", delimiter=",")
print("Saved to predictions.csv")
print("Preview:", ",".join(str(p) for p in predictions[:20]))
