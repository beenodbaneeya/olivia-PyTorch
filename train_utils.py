import torch

def train(model, optimizer, train_loader, loss_fn, device):
    """
    Trains the model for one epoch.Note that, this function will be used only for single gpu implementation. For the multi-gpu implementation, we will be defining the train function in the train_ddp.py file itself.
    Args:
        model(torch.nn.Module): The model to train.
        optimizer(torch.optim.Optimizer): Optimizer for updating model parameters.
        train_loader(torch.utils.data.DataLoader): DataLoader for training data.
        loss_fn (torch.nn.Module): Loss function.
        device (torch.device): Device to run training on (CPU or GPU).
    """
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        # Forward passs
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



def test(model, test_loader, loss_fn, device):
    """
    Evaluates the model on the validation dataset.Note that, this function will be used in the multi-gpu implementation aswell.
    Args:
        model(torch.nn.Module): The model to evaluate.
        test_loader (torch.utils.data.DataLoader): DataLoader for validation data.
        loss_fn (torch.nn.Module): Loss function.
        device (torch.device): Device to run evaluation on (CPU or GPU).
    Returns:
        tuple: Validation accuracy and validaiton loss.
    """

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