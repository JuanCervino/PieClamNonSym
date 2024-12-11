import torch
from torch_geometric.data import Data

# Check PyTorch and CUDA versions
print('Hello Worrr!')
print("PyTorch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device count:", torch.cuda.device_count())
    print("CUDA device name:", torch.cuda.get_device_name(0))

# Create a simple PyTorch Geometric data object
edge_index = torch.tensor([[0, 1],
                           [1, 0]], dtype=torch.long)
x = torch.tensor([[-1], [1]], dtype=torch.float)

data = Data(x=x, edge_index=edge_index)

print("PyTorch Geometric Data object:", data)
