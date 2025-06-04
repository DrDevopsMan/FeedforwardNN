
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset
import os

import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

if device.type == 'cuda':
    print(" -> Name:", torch.cuda.get_device_name(0))
    print(" -> Memory Allocated:", round(torch.cuda.memory_allocated(0)/1024**2, 1), "MB")
    print(" -> Memory Cached:   ", round(torch.cuda.memory_reserved(0)/1024**2, 1), "MB")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Safer folder (you can change this to something simple like "C:/MLP/data")
root_dir = os.path.abspath("data")
os.makedirs(root_dir, exist_ok=True)

# --- 1. Define Transforms ---

# For original training data (no deformation)
original_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# For augmented training data (adds deformation)
augmented_transform = transforms.Compose([
    transforms.RandomAffine(
        degrees=15,            # rotate ±15°
        translate=(0.1, 0.1),  # shift ±10%
        scale=(0.9, 1.1),      # zoom ±10%
        shear=10               # shear ±10°
    ),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# For test data — must be untouched!
test_transform = original_transform

# --- 2. Load Datasets ---

# Original training dataset
original_dataset = datasets.MNIST(
    root=root_dir, train=True, download=True, transform=original_transform
)

# Augmented version of same data
augmented_dataset = datasets.MNIST(
    root=root_dir, train=True, download=True, transform=augmented_transform
)

# Concatenate both to double training size
train_dataset = ConcatDataset([original_dataset, augmented_dataset])

# Clean test set
test_dataset = datasets.MNIST(
    root=root_dir, train=False, download=True, transform=test_transform
)

# --- 3. Create DataLoaders ---

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000)


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28*28, 700)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(700, 650)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(650, 650)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(650, 500)
        self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(500, 450)
        self.relu5 = nn.ReLU()
        self.fc6 = nn.Linear(450, 400)
        self.relu6 = nn.ReLU()
        self.fc7 = nn.Linear(400, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)      # Flatten image
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)            # No softmax here; handled by loss
        x = self.relu3(x)
        x = self.fc4(x)  
        x = self.relu4(x)
        x = self.fc5(x)  
        x = self.relu5(x)
        x = self.fc6(x) 
        x = self.relu6(x)
        x = self.fc7(x) 
        return x
    




model = MLP()
model = MLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Training for 5 epochs
for epoch in range(14):
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)


        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()

        # # --- Track gradient magnitudes ---
        # total_grad = 0
        # param_count = 0
        # for param in model.parameters():
        #     if param.grad is not None:
        #         total_grad += param.grad.abs().mean().item()
        #         param_count += 1
        # avg_grad = total_grad / param_count if param_count > 0 else 0
        # print(f"Epoch [{epoch+1}/20], Loss: {loss.item():.4f}, Avg Grad: {avg_grad:.6f}")

        optimizer.step()

    print(f"Epoch [{epoch+1}/14], Loss: {loss.item():.4f}")



correct = 0
total = 0
model.eval()

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")