import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import StepLR

from torchvision import transforms
from cnn_model import CNN
import numpy as np
import os
import random
import pickle
import matplotlib.pyplot as plt
# from skimage.metrics import structural_similarity as ssim

# # Check if GPU is available
# if torch.cuda.is_available():
#     print(f'Using GPU: {torch.cuda.get_device_name()}')
#     device = torch.device('cuda')
# else:
#     print('Using CPU')
#     device = torch.device('cpu')
# # add current path
# os.chdir(os.path.dirname(os.path.abspath(__file__)))

# load dataset
class custom_dataset(Dataset):
    def __init__(self, X, y, transforms = None):
        self.X = X
        self.y = y
        self.transforms = transforms
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        sample, label = self.X[index], self.y[index]
        if self.transforms:
            sample = self.transforms(sample)
        return sample, label
    
    
    
class ToTensor:
    def __call__(self, sample):
        return torch.tensor(sample, dtype=torch.float32)

class Normalize:
    def __call__(self, sample):
        # Normalize the data
        return (sample - sample.mean()) / sample.std()
    
class custom_horizontal_flip:
    """ flips the tensor along the width dimension with a probability p """
    def __call__(self, sample, p: float = 0.5):
        if torch.rand(1).item() < p:
            sample = torch.flip(sample, dims=[-1])  # Flip along the width dimension
        return sample

# custom vertical flip
class custom_vertical_flip:
    """ flips the tensor along the height dimension with a probability p """
    def __call__(self, sample, p: float = 0.5):
        if torch.rand(1).item() < p:
            sample = torch.flip(sample, dims=[-2])  # Flip along the height dimension
        return sample
       
class Trainer():
    def __init__(self, data_path, model_class, batch_size=8, n_epochs=50, lr=1e-4, step_size=20, gamma=0.5):
        # Initialize parameters
        self.data_path = data_path
        self.model_class = model_class
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.lr = lr
        self.step_size = step_size
        self.gamma = gamma
        
        # Check if GPU is available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using {"GPU" if torch.cuda.is_available() else "CPU"}: {torch.cuda.get_device_name() if torch.cuda.is_available() else ""}')
        
        # Prepare dataset
        self.prepare_data()
        
        # Create dataset and dataloader
        self.create_dataloaders()
        
        # Initialize model, loss, optimizer, and scheduler
        self.model = self.model_class().to(self.device)
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        self.scheduler = StepLR(self.optimizer, step_size=self.step_size, gamma=self.gamma)
    
    def prepare_data(self):
        # Load data from pickle file
        with open(self.data_path, 'rb') as f:
            processed_data_values = pickle.load(f)
        
        # Remove timepoints with NaNs from fields
        fields = np.array([item[0] for item in processed_data_values])
        nan_counts = []
        nan_indices = []
        for i in range(len(fields)):
            nan_count = np.isnan(fields[i]).sum()
            nan_counts.append(nan_count)
            if nan_count > 0:
                nan_indices.append(i)
        fields_clean = np.delete(fields, nan_indices, axis=0)
        
        # Prepare input and output sequences
        time_steps = 10
        X, y = [], []
        for i in range(len(fields_clean) - time_steps+1):
            X.append(fields_clean[i:i + time_steps])
            y.append(fields_clean[i + time_steps-1])
        X = np.array(X)
        X = X[:, np.newaxis, :, :, :]
        y = np.array(y)
        y = y[:, np.newaxis, :, :]
        
        # Convert to PyTorch tensors
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
        # Split data into training and validation sets
        split_index = len(self.X) - 60
        self.X_train, self.X_val = self.X[:split_index], self.X[split_index:]
        self.y_train, self.y_val = self.y[:split_index], self.y[split_index:]
        
    def create_dataloaders(self):
        train_transform = transforms.Compose([
            Normalize(),
            custom_horizontal_flip(),
            custom_vertical_flip(),
        ])

        val_transform = transforms.Compose([
            Normalize()
        ])
        # Create datasets
        self.train_dataset = custom_dataset(self.X_train, self.y_train, transforms=train_transform)
        self.val_dataset = custom_dataset(self.X_val, self.y_val, transforms=val_transform)
        
        # Create dataloaders
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        
    def train(self):
        for epoch in range(self.n_epochs):
            self.model.train()
            for i, (X_batch, y_batch) in enumerate(self.train_dataloader):
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                self.optimizer.zero_grad()
                y_pred = self.model(X_batch)
                loss = self.criterion(y_pred, y_batch)
                loss.backward()
                self.optimizer.step()
            print(f'Epoch {epoch + 1}/{self.n_epochs}, Loss: {loss.item()}')
            
            self.model.eval()
            with torch.no_grad():
                val_loss = 0
                for X_batch, y_batch in self.val_dataloader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    y_pred = self.model(X_batch)
                    val_loss += self.criterion(y_pred, y_batch).item()
                print(f'Validation Loss: {val_loss / len(self.val_dataloader)}')
            # scheduler step
            self.scheduler.step()
        # save the model
        torch.save(self.model.state_dict(), 'cnn_model.pth')

def plot_multiple_instances(X_batch, y_batch, y_pred, num_instances=3):
    batch_size = min(3,X_batch.shape[0])
    indices = random.sample(range(batch_size), num_instances)
    fig, axs = plt.subplots(num_instances, 2, figsize=(10, 5 * num_instances))

    for i, idx in enumerate(indices):
        y_instance = y_batch[idx].cpu().numpy()  # True target
        y_pred_instance = y_pred[idx].detach().cpu().numpy()  # Predicted output
        # ssim_index = ssim(y_instance[0], y_pred_instance[0], data_range=y_pred_instance.max() - y_pred_instance.min())

        axs[i, 0].imshow(y_instance[0], cmap='viridis')  # True target
        axs[i, 0].set_title(f'Ground Truth {i + 1}')
        axs[i, 0].axis('off')

        axs[i, 1].imshow(y_pred_instance[0], cmap='viridis')  # Predicted output
        axs[i, 1].set_title(f'Prediction {i + 1}')
        axs[i, 1].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    trainer = Trainer(data_path='../datasets/processed_data_values.pkl', model_class=CNN)
    # trainer.train()
    trainer.model.eval()
    val_loss = 0
    for X_batch, y_batch in trainer.val_dataloader:
        X_batch, y_batch = X_batch.to(trainer.device), y_batch.to(trainer.device)
        y_pred = trainer.model(X_batch)
        loss = trainer.criterion(y_pred, y_batch)
        val_loss += loss.item()
        # Plot the first 5 instances
        plot_multiple_instances(X_batch, y_batch, y_pred)
        break
    
    average_val_loss = val_loss / len(trainer.val_dataloader)
    print(f'Average validation loss: {average_val_loss}')
        
