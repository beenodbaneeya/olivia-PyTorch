# Overview
This script is a single-GPU implementation for training a WideResNet model on the CIFAR-100 dataset.The script includes features such as early stopping based on validation accuracy and logs key metrics like throughput and epoch duration.

## Key Features
1. Single-GPU Training: Optimized for training on a single GPU.
2. Early Stopping: Stops training early if the target accuracy is achieved for a specified number of epochs.
3. CIFAR-100 Dataset: Automatically loads and prepares the CIFAR-100 dataset.
4. Performance Metrics: Logs validation loss, accuracy, and throughput for each epoch.

## Script Breakdown
1. Imports and Configuration

The script begins by importing necessary libraries and modules:

- PyTorch: For model training, optimization, and evaluation.

- Custom Modules: Includes `dataset_utils` for dataset loading, `model` for the WideResNet implementation, `train_utils` for training and evaluation functions, and `device_utils` for device management.

2. Argument Parsing

The script uses argparse to allow users to configure training parameters via command-line arguments:

`--batch-size`: Batch size for training (default: 256).

`--epochs`: Number of training epochs (default: 100).

`--base-lr`: Learning rate (default: 0.01).

`--target-accuracy`: Target validation accuracy for early stopping (default: 0.95).

`--patience`: Number of consecutive epochs meeting the target accuracy before stopping (default: 2).

````python
parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=256, help='Batch size for training')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
parser.add_argument('--base-lr', type=float, default=0.01, help='Learning rate')
parser.add_argument('--target-accuracy', type=float, default=0.95, help='Target accuracy to stop training')
parser.add_argument('--patience', type=int, default=2, help='Number of epochs that meet target before stopping')
args = parser.parse_args()
````

3. Main Function

The `main` function orchestrates the training process. Below are the key steps:

a. Device Setup:

The script determines the available device (GPU or CPU) using the `get_device` function from `device_utils``

````python
device = get_device()
print(f"Training WideResNet on CIFAR-100 with Batch Size: {args.batch_size}")
````

b. Dataset Loading:

The CIFAR-100 dataset is loaded using the load_cifar100 function. The training and testing datasets are prepared with the specified batch size.

`train_loader, test_loader = load_cifar100(batch_size=args.batch_size)`

c. Model Initialization:

A WideResNet model is initialized with 100 output classes (corresponding to CIFAR-100) and moved to the selected device.

`model = WideResNet(num_classes=100).to(device)`

d. Loss Function and Optimizer:

The loss function is set to CrossEntropyLoss, and the optimizer is SGD with a configurable learning rate.

````python
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.base_lr)
````

4.  Training Loop

The training loop iterates over the specified number of epochs. Key steps include:

a. Epoch Initialization:

The start time for the epoch is recorded to calculate the duration later.

`t0 = time.time()`

b. Training for One Epoch:

The `train_one_epoch` function (from `train_utils`) is called to train the model for one epoch. This function handles the forward pass, backward pass, and optimization for each batch in the training dataset.

`train_one_epoch(model, optimizer, train_loader, loss_fn, device)`

c. Epoch Metrics:

- The epoch duration is calculated, and throughput (images processed per second) is computed.

- The total number of processed images and total training time are updated.

````python
epoch_time = time.time() - t0
images_per_sec = len(train_loader) * args.batch_size / epoch_time
total_images += len(train_loader) * args.batch_size
total_time += epoch_time
````

d. Validation:

The evaluate function (from `train_utils`) is called to compute validation accuracy and loss on the test dataset.

````python
v_accuracy, v_loss = evaluate(model, test_loader, loss_fn, device)
val_accuracy.append(v_accuracy)
````

e. Logging:

Metrics such as epoch duration, validation loss, accuracy, and throughput are logged to the console.

````python
print(f"Epoch {epoch + 1}/{args.epochs}: Time={epoch_time:.3f}s, "
      f"Loss={v_loss:.4f}, Accuracy={v_accuracy:.4f}, "
      f"Throughput={images_per_sec:.1f} img/s")
````

f. Early Stopping:

If the validation accuracy meets or exceeds the target accuracy for the specified number of consecutive epochs (patience), training stops early.

````python
if len(val_accuracy) >= args.patience and all(
    acc >= args.target_accuracy for acc in val_accuracy[-args.patience:]
):
    print(f"Target accuracy reached. Early stopping after epoch {epoch + 1}.")
    break
````

5. Final Summary:

After training is complete, the script logs the final validation accuracy, total training time, and overall throughput.

````python
throughput = total_images / total_time
print(f"\nTraining complete. Final Accuracy: {val_accuracy[-1]:.4f}")
print(f"Total Time: {total_time:.1f}s, Throughput: {throughput:.1f} img/s")
````

