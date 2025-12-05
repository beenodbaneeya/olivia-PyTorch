import os
import time
import argparse
import torch
import torch.nn as nn
import torchvision
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from dataset_utils import load_cifar100
from model import WideResNet
from train_utils import test

#Parse input arguments
parser = argparse.ArgumentParser(description='CIFAR-100 DDP example with Mixed Precision',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--batch-size', type=int, default=512, help='Input batch size for training')
parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train')
parser.add_argument('--base-lr', type=float, default=0.01, help='Learning rate for single GPU')
parser.add_argument('--target-accuracy', type=float, default=0.85, help='Target accuracy to stop training')
parser.add_argument('--patience', type=int, default=2, help='Number of epochs that meet target before stopping')
args = parser.parse_args()

def ddp_setup():
    """Set up the distributed environment."""
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def main_worker():
    ddp_setup()

   # Get the local rank and device
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device(f"cuda:{local_rank}")  #note: we haven´t used get_device from device_utils here

   # Log initialization info
    if global_rank == 0:
        print(f"Training started with {world_size} processes across {world_size // torch.cuda.device_count()} nodes.")
        print(f"Using {torch.cuda.device_count()} GPUs per node.")

    # Load the CIFAR-100 dataset with DistributedSampler
    per_gpu_batch_size = args.batch_size // world_size  # Divide global batch size across GPUs
    train_sampler = DistributedSampler(
        torchvision.datasets.CIFAR100(
            #note: replace this path with the one where you place your datasets folder.
            root="/cluster/work/projects/<project_number>/binod/olivia/datasets/",
            train=True,
            download=True
        )
    )
    train_loader, test_loader = load_cifar100(
        batch_size=per_gpu_batch_size,
        num_workers=8,
        sampler=train_sampler
    )

    # Create the model and wrap it with DDP
    num_classes = 100  # CIFAR-100 has 100 classes
    model = WideResNet(num_classes).to(device)
    model = DDP(model, device_ids=[local_rank])

    # Define loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=5e-4)

    # Initialize gradient scaler for mixed precision
    scaler = torch.cuda.amp.GradScaler()
    val_accuracy = []
    total_time = 0
    total_images = 0  # Total images processed globally

    # Training loop
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)  # Set the sampler epoch for shuffling
        model.train()
        t0 = time.time()

       # Train the model for one epoch
        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

           # Zero the gradients
            optimizer.zero_grad()

           # Forward pass with mixed precision
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = loss_fn(outputs, labels)

           # Backward pass and optimization with scaled gradients
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # Synchronize all processes
        torch.distributed.barrier()
        epoch_time = time.time() - t0
        total_time += epoch_time

        # Compute throughput (images per second for this epoch)
        images_per_sec = len(train_loader) * args.batch_size / epoch_time
        total_images += len(train_loader) * args.batch_size

        # Compute validation accuracy and loss
        v_accuracy, v_loss = test(model, test_loader, loss_fn, device)

        # Average validation metrics across all GPUs
        v_accuracy_tensor = torch.tensor(v_accuracy).to(device)
        v_loss_tensor = torch.tensor(v_loss).to(device)
        torch.distributed.all_reduce(v_accuracy_tensor, op=torch.distributed.ReduceOp.AVG)
        torch.distributed.all_reduce(v_loss_tensor, op=torch.distributed.ReduceOp.AVG)

        # Print metrics only from the main process
        if global_rank == 0:
            print(f"Epoch {epoch + 1}/{args.epochs} completed in {epoch_time:.3f} seconds")
            print(f"Validation Loss: {v_loss_tensor.item():.4f}, Validation Accuracy: {v_accuracy_tensor.item():.4f}")
            print(f"Epoch Throughput: {images_per_sec:.3f} images/second")

        # Early stopping
        val_accuracy.append(v_accuracy_tensor.item())
        if len(val_accuracy) >= args.patience and all(acc >= args.target_accuracy for acc in val_accuracy[-args.patience:]):
            if global_rank == 0:
                print(f"Target accuracy reached. Early stopping after epoch {epoch + 1}.")
            break

    # Log total training time and summary
    if global_rank == 0:
        throughput = total_images / total_time
        print("\nTraining Summary:")
        print(f"Total training time: {total_time:.3f} seconds")
        print(f"Throughput: {throughput:.3f} images/second")
        print(f"Number of nodes: {world_size // torch.cuda.device_count()}")
        print(f"Number of GPUs per node: {torch.cuda.device_count()}")
        print(f"Total GPUs used: {world_size}")
        print("Training completed successfully.")

    # Clean up the distributed environment
    destroy_process_group()
if __name__ == '__main__':
    main_worker()