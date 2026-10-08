import torch
from memory_utils import (
    cleanup,
    format_bytes,
    print_memory
)
DEVICE="mps"
class KVCache:
    def __init__(self,num_layers):
        self.keys=[
            [] for _ in range(num_layers)
        ]
        self.values=[
            [] for _ in range(num_layers)
        ]
    def append(
        self,
        key,
        value,
    ):
        for layer in range(len(self.keys)):
            self.keys[layer].append(key)
            self.values[layer].append(value)
    
    def memory_bytes(self):
        total=0
        for layer in range(len(self.keys)):
            for key in self.keys[layer]:
                total+=(
                    key.numel()*key.element_size()
                )
            for value in self.values[layer]:
                total+=(
                    value.numel()*value.element_size()
                )
        return total

def main():
    cleanup()
    num_layers=6
    num_heads=8
    head_dim=64
    cache=KVCache(num_layers=num_layers)
    print_memory("START")
    for token in range(1,1001):
        key=torch.randn(
            1,num_heads,1,head_dim, dtype=torch.float16,device=DEVICE
        )
        value=torch.randn(
            1,num_heads,1,head_dim,dtype=torch.float16,device=DEVICE
        )
        cache.append(key,value)
        del key
        del value

        if token%100==0:
            print(
                f"Token:{token:4d}  |   "
                f"KV memory: "
                f"{format_bytes(cache.memory_bytes())}"
            )
    print_memory(
        "AFTER 1000 TOKENS"
    )

if __name__=="__main__":
    main()