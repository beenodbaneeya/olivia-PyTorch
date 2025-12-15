# Overview
This script provides utility functions for training and evaluating a PyTorch model. These functions are designed to simplify the training and validation process by handling the core logic for forward and backward passes, loss computation, and accuracy calculation.

The script includes:
- train: A function to train the model for one epoch (single-GPU implementation).
- test: A function to evaluate the model on a validation dataset (used for both single-GPU and multi-GPU implementations)

## Key Functions

1. train

Purpose:
Trains the model for one epoch by performing the following steps:
- Processes batches of training data.
- Computes the loss using the specified loss function.
- Performs backpropagation to compute gradients.
- Updates the model parameters using the optimizer.

Parameters:
- model `(torch.nn.Module)`: The model to be trained.
- optimizer `(torch.optim.Optimizer)`: Optimizer for updating model parameters.
- train_loader `(torch.utils.data.DataLoader)`: DataLoader for the training dataset.
- loss_fn `(torch.nn.Module)`: Loss function to compute the error.
- device `(torch.device)`: Device to run the training on (e.g., CPU or GPU).

Implementation:
- Sets the model to training mode using `model.train()`.
- Iterates over batches of training data from the train_loader.
- Moves the images and labels to the specified device (CPU or GPU).
- Performs a forward pass to compute the model's predictions.
- Computes the loss between predictions and ground truth labels.
- Performs a backward pass to compute gradients.
- Updates the model parameters using the optimizer.

````python
def train(model, optimizer, train_loader, loss_fn, device):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        # Forward pass
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
````

This function is specifically designed for single-GPU training.For multi-GPU training, a separate implementation is defined in the `ddp_train.py` file.

2. test

Purpose:
- Evaluates the model on a validation dataset by:
- Computing the loss on the validation data.
- Calculating the accuracy of the model's predictions.

Parameters:
- model (`torch.nn.Module`): The model to be evaluated.

- test_loader (`torch.utils.data.DataLoader`): DataLoader for the validation dataset.

- loss_fn (`torch.nn.Module`): Loss function to compute the error.

- device (`torch.device`): Device to run the evaluation on (e.g., CPU or GPU).

Returns:
- v_accuracy (float): Validation accuracy (ratio of correct predictions to total samples).

- v_loss (float): Average validation loss across all batches.

Implementation:
- Sets the model to evaluation mode using `model.eval()`.
- Disables gradient computation using `torch.no_grad()` to save memory and improve performance.
- Iterates over batches of validation data from the `test_loader`.
- Moves the images and labels to the specified device (CPU or GPU).
- Performs a forward pass to compute the model's predictions.
- Computes the loss between predictions and ground truth labels.
- Calculates the number of correct predictions and accumulates the total loss.

````python
def test(model, test_loader, loss_fn, device):
    model.eval()
    total_labels = 0
    correct_labels = 0
    loss_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            # Forward pass
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            # Compute accuracy and loss
            predictions = torch.max(outputs, 1)[1]
            total_labels += len(labels)
            correct_labels += (predictions == labels).sum().item()
            loss_total += loss.item()
    v_accuracy = correct_labels / total_labels
    v_loss = loss_total / len(test_loader)
    return v_accuracy, v_loss
````

This function is compatible with both single-GPU and multi-GPU implementations.The accuracy is calculated as the ratio of correct predictions to the total number of labels.The loss is averaged across all batches in the validation dataset.

