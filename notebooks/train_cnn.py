import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import StepLR

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
       

train_transform = transforms.Compose([
    Normalize(),
    custom_horizontal_flip(),
    custom_vertical_flip(),
])

val_transform = transforms.Compose([
    Normalize()
])



# load data from pickle file
import pickle
with open('../datasets/processed_data_values.pkl', 'rb') as f:
    processed_data_values = pickle.load(f)
    
# the shape of my processed data is (num_samples, 3, 100, 100), where 3 is fields, xx, yy
# 100 by 100 is the image size

# the input should be (number_of_sequences, number_of_channels, time_steps, height, width)  

fields = np.array([item[0] for item in processed_data_values])  # shape (729, 100, 100)

nan_counts = []
nan_indices = []

for i in range(len(fields)):
    nan_count = np.isnan(fields[i]).sum()
    nan_counts.append(nan_count)
    if nan_count > 0:
        nan_indices.append(i)

# remove the timepoints with NaNs from fields
fields_clean = np.delete(fields, nan_indices, axis=0)


time_steps = 10 
X,y = [], []

for i in range(len(fields_clean) - time_steps):
    X.append(fields_clean[i:i + time_steps])
    y.append(fields_clean[i + time_steps])

X = np.array(X)  # Shape: (number_of_sequences, time_steps, 100, 100)
X = X[:, np.newaxis, :, :, :]  # Reshape to (number_of_sequences, 1, time_steps, 100, 100) # add 1 for channels
y = np.array(y)  # Shape: (number_of_sequences, 100, 100)
y = y[:, np.newaxis, :, :]  # Reshape to (number_of_sequences, 1, 100, 100)


# Assuming X and y are NumPy arrays
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# test train split
split_index = len(X) -60
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]


# Create dataset
train_dataset = custom_dataset(X_train, y_train, transforms=train_transform)
val_dataset = custom_dataset(X_test, y_test, transforms=val_transform)


# Create dataloader
batch_size = 8  # Adjust as needed
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# import model
model = CNN().to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

scheduler = StepLR(optimizer, step_size=20, gamma=0.5)

# Train the model
n_epochs = 100
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