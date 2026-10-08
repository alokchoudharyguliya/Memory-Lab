# LEARN ABOUT MPS(Metal Performance Shaders)

import torch
from memory_utils import (
    cleanup,
    describe_tensor,
    print_memory
)
def main():
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available on this MAC")
    
    print_memory("START")
    print("\nCreating CPU tensor....")
    
    cpu_tensor=torch.zeros(
        (4096,4096),
        dtype=torch.float32,
        device="cpu"
    )
    describe_tensor(
        "CPU tensor",
        cpu_tensor
    )
    print_memory("\nMoving tensor to MPS....")
    mps_tensor=cpu_tensor.to("mps")
    describe_tensor("MPS tensor",mps_tensor)
    
    print("AFTER CPU -> MPS")
    print(
        "\n Both tensors are still referenced"
    )
    print_memory("BOTH ALIVE")
    print("\nDeleting CPU tensor....")
    del cpu_tensor
    cleanup()
    
    print_memory("\nDeleting MPS tensor....")
    del mps_tensor
    
    cleanup()
    print_memory("AFTER DELETING MPS TENSOR")
    
    
if __name__=="__main__":
    main()