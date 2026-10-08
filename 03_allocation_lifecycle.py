import time
import torch
from memory_utils import (
    cleanup,
    format_bytes,
    print_memory
)

SIZE=4096
def allocate_tensor():
    return torch.randn(
        SIZE, SIZE, dtype=torch.float32,
        device="mps"
    )
    
def main():
    print_memory("START")
    print("\nAllocating tensor....")
    tensor=allocate_tensor()
    print_memory("AFTER ALLOCATION")
    print("\n Performing computation....")
    result=tensor*2.0
    print_memory("AFTER COMPUTATION")
    print("\nDeleting original tensor.....")
    del tensor
    cleanup()
    print_memory("AFTER DELETING ORIGINAL")
    print("\nDeleting result.....")
    del result
    cleanup()
    print_memory("AFTER DELETING RESULT")
    print("\nRepeated allocation experiment")
    for i in range(10):
        tensor=allocate_tensor()
        print_memory(
            f"Iteration {i+1}"
        )
        del tensor
        cleanup()
        time.sleep(0.1)
        
    print("\nClone experiment")
    x=torch.zeros((4096,4096),dtype=torch.float32,device="mps")
    print_memory("X")
    y=x.clone()
    print_memory("X+Y")
    del x
    cleanup()
    print_memory("AFTER DELETING X")
    del y
    cleanup()
    print_memory("AFTER DELETING Y")
    
# y=x.clone() vs y=x  => the latter does not create another tensor storage, that distinction becomes important in memory optimization
# MPS memory -> tensors
#            -> buffers
# each tensor has:
# shape
# dtype
# device
# number of elements
# bytes per element
# total storage
# lifetime
# 
#  
    
if __name__=="__main__":
    main()