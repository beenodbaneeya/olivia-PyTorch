import torchvision
import torchvision.transforms as transforms
import torch
from pathlib import Path
import os

def _data_dir_default():
    repo_root = Path(__file__).resolve().parent
    data_dir = repo_root / "datasets"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def load_cifar100(batch_size, num_workers=0,sampler=None, data_dir=None):
    """
    Loads the CIFAR-100 dataset.Create the dataset directory to store dataset during runtime and no environment variable support.
    """
    root = Path(data_dir).expanduser().resolve() if data_dir else _data_dir_default()

    # Define transformations
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)) # CIFAR-100 mean and std
    ])

    # Load full datasets
    train_set = torchvision.datasets.CIFAR100(
            root=str(root),download=True,train=True,transform=transform)
    test_set = torchvision.datasets.CIFAR100(root=str(root),download=True,train=False,transform=transform)

    # Create the data loaders
    train_loader = torch.utils.data.DataLoader(train_set,batch_size=batch_size,drop_last=True,shuffle=(sampler is None),sampler= sampler,num_workers=num_workers,pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set,batch_size=batch_size,drop_last=True,shuffle=False,num_workers=num_workers,pin_memory=True)
    return train_loader, test_loader