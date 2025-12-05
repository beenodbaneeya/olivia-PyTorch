import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import sys

# Import custom modules
from dataset_utils import load_cifar100
from model  import WideResNet
from train_utils import train as train_one_epoch, test as evaluate
from device_utils import get_device


# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.01
TARGET_ACCURACY = 0.95
PATIENCE = 2

def run_train(batch_size, epochs, learning_rate, device):
    """
    Trains a WideResNet model on the CIFAR-100 dataset for single GPU.
    Args:
        batch_size (int): Batch size for training.
        epochs (int): Number of epochs to train.
        learning_rate (float): Learning rate for the optimizer.
        device (torch.device): Device to run training on (CPU or GPU).
    Returns:
        throughput (float): Images processed per second
    """

    print(f"Training WideResNet on CIFAR-100  with Batch Size: {batch_size}")

    # Training variables
    val_accuracy = []
    total_time = 0
    total_images = 0 # total images processed

    # Load the dataset
    train_loader, test_loader = load_cifar100(batch_size=batch_size)

    # Initialize the WideResNet Model
    num_classes = 100    # num_class set to  100 for CIFAR-100.
    model = WideResNet(num_classes).to(device)

    # Define the loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        t0 = time.time()

        # Train the model for one epoch
        train_one_epoch(model, optimizer, train_loader, loss_fn, device)

        # Calculate epoch time
        epoch_time = time.time() - t0
        total_time += epoch_time

        # Compute throughput (images per second)
        images_per_sec = len(train_loader) * batch_size / epoch_time
        total_images += len(train_loader) * batch_size

        # Compute validation accuracy and loss
        v_accuracy, v_loss = evaluate(model, test_loader, loss_fn, device)
        val_accuracy.append(v_accuracy)

        # Print metrics
        print("Epoch = {:2d}: Epoch Time = {:5.3f}, Validation Loss = {:5.3f}, Validation Accuracy = {:5.3f}, Images/sec = {:5.3f}, Cumulative Time = {:5.3f}".format(
            epoch + 1, epoch_time, v_loss, v_accuracy, images_per_sec, total_time
        ))

        # Early stopping
        if len(val_accuracy) >= PATIENCE and all(acc >= TARGET_ACCURACY for acc in val_accuracy[-PATIENCE:]):
            print('Early stopping after epoch {}'.format(epoch + 1))
            break

    # Final metrics
    throughput = total_images / total_time
    print("\nTraining complete. Final Validation Accuracy = {:5.3f}".format(val_accuracy[-1]))
    print("Total Training Time: {:5.3f} seconds".format(total_time))
    print("Throughput: {:5.3f} images/second".format(throughput))
    return throughput

def main():

    # Set the compute device
    device = get_device()

    # Train the WideResNet model to get the throughput(i.e. images processed per second)
    throughput = run_train(batch_size=BATCH_SIZE, epochs=EPOCHS, learning_rate=LEARNING_RATE, device=device)
    print(f"Single-GPU Thrpughput: {throughput:.3f} images/second")

if __name__ == "__main__":
    main()
