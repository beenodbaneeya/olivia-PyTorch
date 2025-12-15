import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim

from dataset_utils import load_cifar100
from model import WideResNet
from train_utils import train as train_one_epoch, test as evaluate
from device_utils import get_device

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=256, help='Batch size for training')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
parser.add_argument('--base-lr', type=float, default=0.01, help='Learning rate')
parser.add_argument('--target-accuracy', type=float, default=0.95, help='Target accuracy to stop training')
parser.add_argument('--patience', type=int, default=2, help='Number of epochs that meet target before stopping')
args = parser.parse_args()

def main():
    device = get_device()
    print(f"Training WideResNet on CIFAR-100 with Batch Size: {args.batch_size}")

    # Training variables
    val_accuracy = []
    total_time = 0
    total_images = 0

    # Load the dataset
    train_loader, test_loader = load_cifar100(batch_size=args.batch_size)

    # Initialize the model
    model = WideResNet(num_classes=100).to(device)

    # Define loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.base_lr)

    for epoch in range(args.epochs):
        t0 = time.time()

        # Train for one epoch
        train_one_epoch(model, optimizer, train_loader, loss_fn, device)

        epoch_time = time.time() - t0
        total_time += epoch_time

        # Compute throughput
        images_per_sec = len(train_loader) * args.batch_size / epoch_time
        total_images += len(train_loader) * args.batch_size

        # Evaluate
        v_accuracy, v_loss = evaluate(model, test_loader, loss_fn, device)
        val_accuracy.append(v_accuracy)

        print(f"Epoch {epoch + 1}/{args.epochs}: Time={epoch_time:.3f}s, "
              f"Loss={v_loss:.4f}, Accuracy={v_accuracy:.4f}, "
              f"Throughput={images_per_sec:.1f} img/s")

        # Early stopping
        if len(val_accuracy) >= args.patience and all(
            acc >= args.target_accuracy for acc in val_accuracy[-args.patience:]
        ):
            print(f"Target accuracy reached. Early stopping after epoch {epoch + 1}.")
            break

    # Final summary
    throughput = total_images / total_time
    print(f"\nTraining complete. Final Accuracy: {val_accuracy[-1]:.4f}")
    print(f"Total Time: {total_time:.1f}s, Throughput: {throughput:.1f} img/s")

if __name__ == "__main__":
    main()