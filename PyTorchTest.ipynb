'''Setting up the data'''
from sklearn.model_selection import train_test_split
import torch

# Extract features (X) and target variable (y)
X = df_demand[['hour', 'dayofweek', 'quarter', 'month','year','dayofyear','dayofmonth', 'weekofyear']]  # Include your features here
y = df_demand['Actual Total Load [MW] - BZN|NO1']  # Assuming 'Actual Total Load [MW] - BZN|NO1' is your target variable

# Convert non-numeric columns to numeric types if needed
X = X.astype(float)  # Convert X features to float type if they're not numeric

# Split the data into training and test sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)  # Avoid shuffling time series data

# Convert Pandas DataFrames to PyTorch tensors
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

# Check the shapes of the tensors
print(f"X_train shape: {X_train_tensor.shape}, y_train shape: {y_train_tensor.shape}")
print(f"X_test shape: {X_test_tensor.shape}, y_test shape: {y_test_tensor.shape}")


import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Define your PyTorch model for multi-output regression
class YourModel(nn.Module):
    def __init__(self, input_size):
        super(YourModel, self).__init__()
        self.hidden = nn.Linear(input_size, 128)  # Hidden layer
        self.output_layer = nn.Linear(128, 7008)  # Output layer with 7008 units

    def forward(self, x):
        x = torch.relu(self.hidden(x))
        x = self.output_layer(x)
        return x

'''# Assuming X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor are defined
# Normalizing input features (X)
mean = X_train_tensor.mean(dim=0)
std = X_train_tensor.std(dim=0)
X_train_normalized = (X_train_tensor - mean) / std
X_test_normalized = (X_test_tensor - mean) / std'''

# Initialize your model
input_size = X_train_tensor.shape[1]  # Assuming input size matches number of features
model = YourModel(input_size)

# Define loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop example
num_epochs = 10
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)  # Ensure targets have proper shape
    loss.backward()
    optimizer.step()

    # Optionally print loss after each epoch
    if (epoch+1) % 1 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
