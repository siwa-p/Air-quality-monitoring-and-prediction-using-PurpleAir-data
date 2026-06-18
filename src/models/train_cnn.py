import random
import pickle

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import StepLR

from src.models.cnn_model import CNN


class _Dataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        sample, label = self.X[index], self.y[index]
        if self.transform:
            sample = self.transform(sample)
        return sample, label


class _Normalize:
    def __call__(self, sample):
        return (sample - sample.mean()) / sample.std()


class _HFlip:
    def __call__(self, sample):
        if torch.rand(1).item() < 0.5:
            sample = torch.flip(sample, dims=[-1])
        return sample


class _VFlip:
    def __call__(self, sample):
        if torch.rand(1).item() < 0.5:
            sample = torch.flip(sample, dims=[-2])
        return sample


def _compose(*transforms):
    def apply(x):
        for t in transforms:
            x = t(x)
        return x
    return apply


class Trainer:
    def __init__(self, data_path, model_class=CNN, batch_size=8, n_epochs=50,
                 lr=1e-4, step_size=20, gamma=0.5):
        self.data_path = data_path
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'Device: {"GPU: " + torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU"}')

        self._prepare_data()
        self._create_dataloaders()

        self.model = model_class().to(self.device)
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.scheduler = StepLR(self.optimizer, step_size=step_size, gamma=gamma)

    def _prepare_data(self):
        with open(self.data_path, "rb") as f:
            raw = pickle.load(f)

        fields = np.array([item[0] for item in raw])
        nan_mask = np.array([np.isnan(f).sum() > 0 for f in fields])
        fields = fields[~nan_mask]

        T = 10
        X = np.array([fields[i:i + T] for i in range(len(fields) - T + 1)])
        y = np.array([fields[i + T - 1] for i in range(len(fields) - T + 1)])
        X = X[:, np.newaxis, :, :, :]
        y = y[:, np.newaxis, :, :]

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        split = len(self.X) - 60
        self.X_train, self.X_val = self.X[:split], self.X[split:]
        self.y_train, self.y_val = self.y[:split], self.y[split:]

    def _create_dataloaders(self):
        train_transform = _compose(_Normalize(), _HFlip(), _VFlip())
        val_transform = _Normalize()
        self.train_dataloader = DataLoader(
            _Dataset(self.X_train, self.y_train, train_transform),
            batch_size=self.batch_size, shuffle=True,
        )
        self.val_dataloader = DataLoader(
            _Dataset(self.X_val, self.y_val, val_transform),
            batch_size=self.batch_size, shuffle=False,
        )

    def train(self):
        for epoch in range(self.n_epochs):
            self.model.train()
            epoch_loss = 0.0
            for X_batch, y_batch in self.train_dataloader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                self.optimizer.zero_grad()
                loss = self.criterion(self.model(X_batch), y_batch)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            self.model.eval()
            with torch.no_grad():
                val_loss = sum(
                    self.criterion(self.model(xb.to(self.device)), yb.to(self.device)).item()
                    for xb, yb in self.val_dataloader
                )
            print(
                f"Epoch {epoch+1}/{self.n_epochs}  "
                f"Train Loss: {epoch_loss/len(self.train_dataloader):.4f}  "
                f"Val Loss: {val_loss/len(self.val_dataloader):.4f}"
            )
            self.scheduler.step()

        torch.save(self.model.state_dict(), "datasets/cnn_model.pth")


def plot_predictions(X_batch, y_batch, y_pred, n=3):
    indices = random.sample(range(min(3, X_batch.shape[0])), n)
    fig, axs = plt.subplots(n, 2, figsize=(10, 5 * n))
    for i, idx in enumerate(indices):
        axs[i, 0].imshow(y_batch[idx].cpu().numpy()[0], cmap="viridis")
        axs[i, 0].set_title(f"Ground Truth {i + 1}")
        axs[i, 0].axis("off")
        axs[i, 1].imshow(y_pred[idx].detach().cpu().numpy()[0], cmap="viridis")
        axs[i, 1].set_title(f"Prediction {i + 1}")
        axs[i, 1].axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    trainer = Trainer(data_path="datasets/processed_data_values.pkl")
    trainer.train()
