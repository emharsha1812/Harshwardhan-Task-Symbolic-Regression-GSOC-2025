import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

# Define a simple toy model.
class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)

def train(rank, world_size):
    setup(rank, world_size)
    
    # Explicitly set the current device for this process.
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    
    # Print which GPU is being used by this process.
    print(f"Rank {rank} is using GPU {torch.cuda.current_device()} ({torch.cuda.get_device_name(rank)})")
    
    torch.manual_seed(0)
    # Create a toy dataset: 100 samples with 10 features each.
    x = torch.randn(100, 10)
    y = torch.randn(100, 1)
    dataset = TensorDataset(x, y)
    
    # Use DistributedSampler to partition the dataset among processes.
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=8, sampler=sampler)
    
    # Create and wrap the model.
    model = ToyModel().to(device)
    ddp_model = DDP(model, device_ids=[rank])
    
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    num_epochs = 500
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)  # Shuffle data differently each epoch.
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
    
            optimizer.zero_grad()
            outputs = ddp_model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    
            # Print status to track training progress and GPU usage.
            if batch_idx % 5 == 0:
                print(f"Rank {rank} on GPU {torch.cuda.current_device()}: Epoch {epoch}, Batch {batch_idx}, Loss {loss.item():.4f}")
    
    cleanup()

if __name__ == "__main__":
    # Use the number of available GPUs as the world size.
    world_size = torch.cuda.device_count()
    mp.spawn(train,
             args=(world_size,),
             nprocs=world_size,
             join=True)
