import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from config import BATCH_SIZE
import numpy as np
import pickle
import os

CIFAR_WIDTH = 32
CIFAR_HEIGHT = 32
CIFAR_CHANNEL = 3


default_dataset_path = os.path.join(os.path.dirname(__file__), "dataset")
default_cifar10_path = os.path.join(default_dataset_path, "cifar-10-batches-py")


def dataset_generator(dir, mode=None):
    # Define the transform for the dataset
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    # Create the dataset
    if mode:
        dataset = datasets.ImageFolder(
            root=dir,
            transform=transform
        )
        # Split the dataset into training and validation subsets
        val_size = int(0.1 * len(dataset))
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size],
                                                                   generator=torch.Generator().manual_seed(0))

        if mode == 'training':
            data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        elif mode == 'validation':
            data_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    else:
        dataset = datasets.ImageFolder(
            root=dir,
            transform=transform
        )
        data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    return data_loader


def get_cifar_batch(file_name: str):
    with open(file_name, "rb") as f:
        batch_data = pickle.load(f, encoding="bytes")
        batch_data[b"data"] = batch_data[b"data"] / 255
        return batch_data[b"data"], batch_data[b"labels"]

def get_cifar10_data(
    dataset_path: str = default_cifar10_path,
    num_samples_train: int = 45000,
    num_samples_val: int = 5000,
    shuffle: bool = False,
    return_image: bool = False,
    feature_process: any = None,
    subset_train: int = None,
    subset_val: int = None,
    subset_test: int = None,
):
    x_train = []
    y_train = []
    for i in range(1, 6):
        x_batch, y_batch = get_cifar_batch(os.path.join(dataset_path, "data_batch_{}".format(i)))
        x_train.append(x_batch)
        y_train.append(y_batch)

    x_train = np.concatenate(x_train)
    y_train = np.concatenate(y_train)

    total_idx = np.arange(50000)
    # val_idx = np.random.choice(50000, num_samples_val, replace=False)
    # train_idx = np.delete(total_idx, val_idx)
    # Use Deterministic split version
    train_idx = np.arange(num_samples_train)
    val_idx = np.arange(num_samples_val) + num_samples_train

    x_val = x_train[val_idx]
    y_val = y_train[val_idx]

    x_train = x_train[train_idx]
    y_train = y_train[train_idx]

    assert num_samples_train + num_samples_val == 5 * 10000

    x_test, y_test = get_cifar_batch(os.path.join(dataset_path, "test_batch"))
    y_test = np.array(y_test)

    if subset_train is None:
        subset_train = num_samples_train
    if subset_val is None:
        subset_val = num_samples_val
    if subset_test is None:
        subset_test = 10000
    dataset = {
        "x_train": x_train[:subset_train],
        "y_train": y_train[:subset_train],
        "x_val": x_val[:subset_val],
        "y_val": y_val[:subset_val],
        "x_test": x_test[:subset_test],
        "y_test": y_test[:subset_test],
    }
    if return_image:
        dataset["x_train"] = dataset["x_train"].reshape((-1, CIFAR_CHANNEL, CIFAR_WIDTH, CIFAR_HEIGHT))
        dataset["x_val"] = dataset["x_val"].reshape((-1, CIFAR_CHANNEL, CIFAR_WIDTH, CIFAR_HEIGHT))
        dataset["x_test"] = dataset["x_test"].reshape((-1, CIFAR_CHANNEL, CIFAR_WIDTH, CIFAR_HEIGHT))
    if feature_process is not None:
        import time

        c_t = time.time()
        print("Start Processing")
        dataset["x_train"] = feature_process(dataset["x_train"])
        dataset["x_val"] = feature_process(dataset["x_val"])
        dataset["x_test"] = feature_process(dataset["x_test"])
        print("Processing Time:", time.time() - c_t)
    return dataset



