from torchvision import datasets, transforms
import torch

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomResizedCrop(32),
    transforms.RandomRotation(0.3),
    transforms.ColorJitter(contrast=0.3)
])

train_set = datasets.CIFAR10(root="dataset", train=True, download=True, transform=transform)
test_set = datasets.CIFAR10(root="dataset", train=False, download=True, transform=transforms.ToTensor())

train_set, val_set = torch.utils.data.random_split(train_set, [0.9, 0.1])
