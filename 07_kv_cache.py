import torch
from memory_utils import(
    cleanup,
    describe_tensor,
    format_bytes,
    print_memory
)

# For an attention layer, we calculate:
# Q = XWq
# K = XWk
# V = XWv


# During autoregressive generation:
# Token 1
#    ↓
# K1 V1

# Token 2
#    ↓
# K1 K2
# V1 V2

# Token 3
#    ↓
# K1 K2 K3
# V1 V2 V3

# Instead of recomputing previous K/V values, we keep them.
# That's the KV cache.


# 2. KV-cache memory formula
# For a simplified Transformer:
# KV memory =
# 2 × layers
#   × sequence_length
#   × KV_heads
#   × head_dimension
#   × bytes_per_element

# The 2 is K + V

# For example:
# layers       = 6
# sequence     = 2048
# KV heads     = 8
# head_dim     = 64
# dtype        = FP16 = 2 bytes

# KV cache ∝ sequence length
# Double sequence length 2× tokens
#    ↓
# 2× KV memory
# We aren't copying the tensor
# key variable
#  │
#  └───────┐
#          ↓
#       Tensor
#          ↑
#          │
#    cache reference
# 
# del key -> doesn't necessarily release the underlying storage. The cache still references it.
# 

DEVICE="mps"

class KVCache:
    def __init__(self,num_layers,num_kv_heads,head_dim,dtype=torch.float16,device=DEVICE):
        self.num_layers=num_layers
        self.num_kv_heads=num_kv_heads
        self.head_dim=head_dim
        self.dtype=dtype
        self.device=device

        self.keys=[
            [] for _ in range(num_layers)
        ]
    
    def append(self,layer_idx,key,value,):
        self.keys[layer_idx].append(key)
        self.values[layer_idx].append(value)
    def get(self,layer_idx):
        keys=torch.cat(
            self.keys[layer_idx],
            dim=2,
        )
        values=torch.cat(
            self.values[layer_idx],
            dim=2
        )
        return keys, values
    
    def memory_bytes(self):
        total=0
        for layer in range(self.num_layers):
            for key in self.keys[layer]:
                total+=(
                    key.numel()*key.element_size()
                )
            for value in self.values[layer]:
                total+=(
                    value.numel()*value.element_size()
                )
        return total
    
    def sequence_length(self):
        if not self.keys[0]:
            return 0
        return sum(
            tensor.shape[2] for tensor in self.keys[0]
        )
        
def main():
    cleanup()
    num_layers=6
    num_kv_heads=8
    head_dim=64
    cache=KVCache(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )
    print_memory("START")
    
    key=torch.randn(
        1,num_kv_heads,1,head_dim,dtype=torch.float16,device=DEVICE
    )
    value=torch.randn(
        1,num_kv_heads,1,head_dim,dtype=torch.float16,device=DEVICE
    )
    print("\nKey:")
    describe_tensor("K",key)
    print("\nValue:")
    describe_tensor("V",value)
    for layer in range(num_layers):
        cache.append(layer,key,value)
    print_memory("AFTER CACHE INSERT")
    print(
        "\nCache memory:",
        format_bytes(cache.memory_bytes())
    )
    print("Sequence Length:",cache.sequence_length())
    del key
    del value
    cleanup()
    print_memory("AFTER ORIGINAL K/V DELETE")
    print(
        "\nCache still owns references"
    )
    print(
        "Cache memory",
        format_bytes(cache.memory_bytes())
        )
if __name__=="__main__":
    main()