# Overview
The `train.py` script is designed for distributed training of a WideResNet model on the CIFAR-100 dataset using PyTorch's Distributed Data Parallel (DDP) framework. It incorporates mixed precision training for improved performance and efficiency. The script is highly configurable and supports early stopping based on validation accuracy.

# Key Features
1. Distributed Training: Utilizes PyTorch's DDP for efficient multi-GPU training.

2. Mixed Precision: Leverages `torch.amp` for faster training with reduced memory usage.

3. Early Stopping: Stops training early if the target accuracy is achieved for a specified number of epochs.

4. CIFAR-100 Dataset: Automatically downloads and prepares the CIFAR-100 dataset.

5. Performance Metrics: Logs throughput, validation loss, and accuracy for each epoch.

# Scripts Breakdown
1.  Imports and Configuration

The script begins by importing necessary libraries and setting up configurations.

- Libraries: Includes PyTorch, torchvision, and custom utility modules (`dataset_utils`, `train_utils`, and `model`).

- Dataset Directory: The dataset is downloaded to the `./datasets` directory by default.

`DATA_DIR = "./datasets"`

2. Argument Parsing

The script uses `argparse` to allow users to configure training parameters via command-line arguments:

`--batch-size`: Batch size for training (default: 512).

`--epochs`: Number of training epochs (default: 5).

`--base-lr`: Learning rate for a single GPU (default: 0.01).

`--target-accuracy`: Target validation accuracy for early stopping (default: 0.85).

`--patience`: Number of consecutive epochs meeting the target accuracy before stopping (default: 2).

````python
parser = argparse.ArgumentParser(description='CIFAR-100 DDP example with Mixed Precision')
parser.add_argument('--batch-size', type=int, default=512, help='Input batch size for training')
parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train')
parser.add_argument('--base-lr', type=float, default=0.01, help='Learning rate for single GPU')
parser.add_argument('--target-accuracy', type=float, default=0.85, help='Target accuracy to stop training')
parser.add_argument('--patience', type=int, default=2, help='Number of epochs that meet target before stopping')
args = parser.parse_args()
````

3. Distributed Setup

The `ddp_setup` function initializes the distributed training environment using the NCCL backend. It also sets the local GPU device for each process.

````python
def ddp_setup():
    """Set up the distributed environment."""
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
````

4. Main Training Workflow

The `main_worker` function orchestrates the entire training process. Below are the key steps:

A. Initialization

- Retrieves the local rank, global rank, and world size from environment variables.
- Logs training information (e.g., number of GPUs and nodes) from the main process.

````python
local_rank = int(os.environ["LOCAL_RANK"])
global_rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
device = torch.device(f"cuda:{local_rank}")
````

B. Dataset Loading

- The CIFAR-100 dataset is loaded using a DistributedSampler to ensure proper data distribution across GPUs.
- The global batch size is divided among GPUs.

````python
per_gpu_batch_size = args.batch_size // world_size
train_sampler = DistributedSampler(
    torchvision.datasets.CIFAR100(
        root=DATA_DIR,
        train=True,
        download=True
    )
)
train_loader, test_loader = load_cifar100(
    batch_size=per_gpu_batch_size,
    num_workers=8,
    sampler=train_sampler
)
````

C. Model and Optimizer Setup

- A WideResNet model is created and wrapped with DDP for distributed training.
- The loss function is set to CrossEntropyLoss, and the optimizer is SGD with momentum and weight decay.

````python
model = WideResNet(num_classes).to(device)
model = DDP(model, device_ids=[local_rank])
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=5e-4)
````

D. Mixed Precision Training

A gradient scaler is initialized for mixed precision training using `torch.amp.GradScaler`

`scaler = torch.amp.GradScaler('cuda')`

5. Training Loop

The training loop iterates over the specified number of epochs with these steps.

A. Epoch Initialization

- The DistributedSampler is updated for shuffling.
- Training begins, and the start time is recorded.

````python
train_sampler.set_epoch(epoch)
model.train()
t0 = time.time()
````

B. Batch Processing

- Each batch of images and labels is moved to the GPU.
- Mixed precision is used for the forward pass, and gradients are scaled during the backward pass.

````python
with torch.amp.autocast('cuda'):
    outputs = model(images)
    loss = loss_fn(outputs, labels)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
````

C. Validation and Metrics

- Validation accuracy and loss are computed using the test function.
- Metrics are averaged across all GPUs using torch.distributed.all_reduce

````python
v_accuracy, v_loss = test(model, test_loader, loss_fn, device)
torch.distributed.all_reduce(v_accuracy_tensor, op=torch.distributed.ReduceOp.AVG)
torch.distributed.all_reduce(v_loss_tensor, op=torch.distributed.ReduceOp.AVG)
````

D. Logging and Early Stopping

- Metrics are logged by the main process.
- Training stops early if the target accuracy is achieved for the specified patience.

````python
if len(val_accuracy) >= args.patience and all(acc >= args.target_accuracy for acc in val_accuracy[-args.patience:]):
    if global_rank == 0:
        print(f"Target accuracy reached. Early stopping after epoch {epoch + 1}.")
    break
````

6. Final Logging and Cleanup

- Total training time and throughput are logged.
- The distributed environment is cleaned up using `destroy_process_group`

`destroy_process_group()`

