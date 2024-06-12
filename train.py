import numpy as np
import math
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import time
from config import EPOCHS, BATCH_SIZE
from models.model import E2EImageCommunicator


USE_GPU = True

dtype = torch.float32 # we will be using float throughout this tutorial

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print('using device:', device)

class RandomRGBShift:
    def __init__(self, mean=0.0, std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, img):

        # Add random RGB shift
        shift = np.random.normal(self.mean, self.std, (3, 1, 1))
        img = img + shift

        # Clip values to be in range [0, 1]
        img = np.clip(img, 0, 1)

        return img

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomResizedCrop(32),
    transforms.RandomRotation(0.3),
    transforms.ColorJitter(contrast=0.3),
    RandomRGBShift()
])

train_set = datasets.CIFAR10(root="utils/dataset", train=True, download=True, transform=transform)
test_set = datasets.CIFAR10(root="utils/dataset", train=False, download=True, transform=transforms.ToTensor())

# train_set, val_set = torch.utils.data.random_split(train_set, [0.9, 0.1])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
# val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# Define the model, loss function, and optimizer
model = E2EImageCommunicator(snrdB=25, channel='Rayleigh').to(device)

model.load_state_dict(torch.load('/home/tongqing/ECE285/Project/weights_unet/epoch_299.pth'))

loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5000, T_mult=1, eta_min=0.1)

# Training and testing functions
def train_step(images):
    model.train()
    optimizer.zero_grad()
    predictions = model(images)

    # if torch.isnan(predictions).any():
    #     print("stop")

    loss = loss_function(images, predictions)

    loss.backward()

    optimizer.step()
    return loss.item()

def test_step(images):
    model.eval()
    with torch.no_grad():
        predictions = model(images)
        loss = loss_function(images, predictions)
    return loss.item()

if not os.path.isdir('./weights'):
    os.mkdir('./weights')

# Training loop
lowest_loss = float('inf')

for epoch in range(1, EPOCHS + 1):
    start_time_training = time.time()
    train_loss_total = 0
    test_loss_total = 0

    # Training
    for i, (images, _) in enumerate(train_loader):
        images = images.to(device=device, dtype=dtype)
        loss = train_step(images)

        train_loss_total += loss

        if i == len(train_loader) // 100:
            estimated_time = len(train_loader) * (time.time() - start_time_training) / (i + 1) / 60
            print(f'Estimated train epoch time: {estimated_time:.2f} minutes')

    

    start_time_test = time.time()
    # Testing
    for i, (images, _) in enumerate(test_loader):
        images = images.to(device=device, dtype=dtype)
        loss = test_step(images)
        test_loss_total += loss

        if i == len(test_loader) // 100:
            estimated_time = len(test_loader) * (time.time() - start_time_test) / (i + 1) / 60
            print(f'Estimated test epoch time: {estimated_time:.2f} minutes')

    train_loss_avg = train_loss_total / len(train_loader)
    test_loss_avg = test_loss_total / len(test_loader)

    print(f'Epoch {epoch}, Loss: {train_loss_avg:.6f}, Test Loss: {test_loss_avg:.6f}, '
          f'Training time: {(start_time_test - start_time_training)/60:.2f}m, Learning rate: {scheduler.get_last_lr()[0]}')

    scheduler.step()

    # Save the best model
    if test_loss_avg < lowest_loss:
        lowest_loss = test_loss_avg
        torch.save(model.state_dict(), f'weights/epoch_{epoch}.pth')
