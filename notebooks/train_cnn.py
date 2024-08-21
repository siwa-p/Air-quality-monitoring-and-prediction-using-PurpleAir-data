import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from cnn_model import CNN
import numpy as np
import os

# Check if GPU is available
if torch.cuda.is_available():
    print(f'Using GPU: {torch.cuda.get_device_name()}')
    device = torch.device('cuda')
else:
    print('Using CPU')
    device = torch.device('cpu')
# add current path
os.chdir(os.path.dirname(os.path.abspath(__file__)))

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

transform = transforms.Compose([
    # ToTensor(),
    Normalize()
])



# load data from pickle file
import pickle
with open('processed_data_values.pkl', 'rb') as f:
    processed_data_values = pickle.load(f)

fields = [item[0] for item in processed_data_values] # List of interpolated fields

data_array = np.stack(fields, axis=0)

data_array = data_array[:, np.newaxis, :, :]  # For PyTorch
# data_array = data_array[:, :, :, np.newaxis] 


time_steps = 10
X, y = [], []
for i in range(len(data_array) - time_steps):
    X.append(data_array[i:i + time_steps])
    y.append(data_array[i + time_steps])

X = np.array(X).reshape(-1, 1,time_steps, 100, 100)
y = np.array(y)


# Assuming X and y are NumPy arrays
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# test train split
split_index = len(X) -60
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]


# Create dataset
train_dataset = custom_dataset(X_train, y_train, transforms=transform)
val_dataset = custom_dataset(X_test, y_test, transforms=transform)


# Create dataloader
batch_size = 4  # Adjust as needed
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# import model
model = CNN().to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
n_epochs = 10
for epoch in range(n_epochs):
    model.train()  
    for i, (X_batch, y_batch) in enumerate(train_dataloader):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}/{n_epochs}, Loss: {loss.item()}')
    
    model.eval()
    with torch.inference_mode():
        val_loss = 0
        for X_batch, y_batch in val_dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            val_loss += criterion(y_pred, y_batch).item()
        print(f'Validation Loss: {val_loss / len(val_dataloader)}')

# Save the model
torch.save(model.state_dict(), 'cnn_model.pth')

    
  


    
