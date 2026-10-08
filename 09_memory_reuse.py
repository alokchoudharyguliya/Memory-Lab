import torch
from memory_utils import(
    cleanup,
    format_bytes,
    print_memory
)
DEVICE="mps"

def bad_allocation():
    tensors=[]
    for _ in range(100):
        tensor=torch.zeros(
            1024,1024,dtype=torch.float16, device=DEVICE
        )
        tensors.append(tensor)
    return tensors

def good_reuse():
    buffer=torch.zeros(
        1024,1024,dtype=torch.float16,device=DEVICE
    )
    for _ in range(100):
        buffer.zero_()
        buffer+=1
    return buffer
def main():
    cleanup()
    print("BAD APPROACH")
    print_memory("START")
    tensors=bad_allocation()
    print_memory("AFTER 100 ALLOCATIONS")
    del tensors
    cleanup()
    
    print_memory("AFTER CLEANUP")
    
    print("\nGOOD APPROACH")
    print_memory("START")
    buffer=good_reuse()
    print_memory("AFTER BUFFER REUSE")
    del buffer
    cleanup()
    print_memory("AFTER FINAL CLEANUP")
    
if __name__=="__main__":
    main()
    
    
# allocate buffer -> use -> reset -> use -> reset -> use
# this is basic idea behind preallocation and buffer reuse
# Python object
#       ↓
# Tensor
#       ↓
# PyTorch allocator
#       ↓
# MPS / Metal runtime
#       ↓
# physical unified memory

# When you delete a tensor, memory can become:
# unused by this tensor

# without immediately becoming:
# returned to the operating system



#                 LLM
#                  │
#     ┌────────────┼────────────┐
#     ↓            ↓            ↓
#  Weights     Activations    KV Cache
#     │            │            │
#   fixed       temporary      grows
#                                │
#                                ↓
#                           sequence length


#                    Memory
#                      │
#        ┌─────────────┼─────────────┐
#        ↓             ↓             ↓
#    Precision      Lifetime       Reuse
#        │             │             │
#  FP32→FP16      delete early    preallocate
#        │             │             │
#        └─────────────┴─────────────┘
#                      ↓
#               lower memory usage