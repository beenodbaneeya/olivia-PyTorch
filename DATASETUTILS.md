# Overview
This script provides utility functions for loading the CIFAR-100 dataset in PyTorch. It includes functionality to:

1. Automatically create a dataset directory for storing the CIFAR-100 dataset.

2. Apply data transformations for preprocessing.

3. Create PyTorch DataLoaders for efficient data loading during training and testing.

The script is designed to be modular and supports both single-GPU and distributed training setups.

## Key Functions
1. `_data_dir_default()`

Purpose:

Creates a default directory (datasets) in the root of the repository to store the CIFAR-100 dataset if no custom directory is provided.

Implementation:
- Determines the root directory of the script using `Path(__file__).resolve().parent`.

- Creates a datasets directory if it doesn't already exist using `mkdir(parents=True, exist_ok=True)`.

````python
def _data_dir_default():
    repo_root = Path(__file__).resolve().parent
    data_dir = repo_root / "datasets"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir
````

This function ensures that the dataset is stored locally within the project directory, making it easier to manage and share the codebase.

2. `load_cifar100()`

A. Purpose:

Loads the CIFAR-100 dataset, applies transformations, and returns PyTorch DataLoaders for training and testing.

Parameters:
- `batch_size (int)`: Number of samples per batch.

- `num_workers (int, default=0)`: Number of subprocesses for data loading.

- `sampler (optional)`: A PyTorch sampler for distributed training.

- `data_dir (optional)`: Custom directory to store the dataset. If not provided, the default directory (datasets) is used.

Returns:
- `train_loader`: DataLoader for the training dataset.

- `test_loader`: DataLoader for the testing dataset.

B. Implementation Steps:

Dataset Directory

- If data_dir is provided, it resolves the path and uses it as the dataset directory.

- If not, it calls _data_dir_default() to create and use the default datasets directory

`root = Path(data_dir).expanduser().resolve() if data_dir else _data_dir_default()`

Data Transformations

Applies a series of transformations to preprocess the CIFAR-100 images.

- RandomHorizontalFlip: Randomly flips the image horizontally to augment the dataset.

- RandomCrop: Crops the image to 32x32 pixels with padding of 4 pixels for additional augmentation.

- ToTensor: Converts the image to a PyTorch tensor.

- Normalize: Normalizes the image using the mean and standard deviation of the CIFAR-100 dataset.

````python
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)) # CIFAR-100 mean and std
])
````

Dataset Loading

- Downloads the CIFAR-100 dataset (if not already downloaded) using torchvision.datasets.CIFAR100.

- Applies the defined transformations to both the training and testing datasets.

````python
train_set = torchvision.datasets.CIFAR100(
    root=str(root), download=True, train=True, transform=transform
)
test_set = torchvision.datasets.CIFAR100(
    root=str(root), download=True, train=False, transform=transform
)
````

DataLoaders

Creates PyTorch DataLoaders for both the training and testing datasets.

- Training DataLoader: Uses the provided sampler if specified (e.g., for distributed training).Shuffles the data if no sampler is provided.

- Testing DataLoader: Does not shuffle the data to maintain consistency during evaluation.

````python
train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=batch_size,
    drop_last=True,
    shuffle=(sampler is None),
    sampler=sampler,
    num_workers=num_workers,
    pin_memory=True
)
test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=batch_size,
    drop_last=True,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True
)
````

