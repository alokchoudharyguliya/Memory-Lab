import torch
import torch.nn as nn
from memory_utils import (
    cleanup,
    format_bytes,
    print_memory
)
DEVICE="mps"

# For the same number of parameters:
# FP32 4 bytes
# versus:
# FP16 2 bytes

# So approximately:
# FP16 weight memory ≈ 50% of FP32

# The key thing to notice is that changing precision changes more than just the final model file size.

# It can affect:
#     weights
#     activations
#     temporary tensors
#     bandwidth
#     cache behavior
#     memory pressure

# see two independent dimensions:
#                 Memory
#                   │
#         ┌─────────┴──────────┐
#         │                    │
#      Precision          Sequence length
#         │                    │
#      FP32/16              128/512/2048
#         │                    │
#         └─────────┬──────────┘
#                   ↓
#              Memory usage



class SmallModel(nn.Module):
    def __init__(
        self,
        vocab_size=10_000,
        d_model=512,
        n_layers=6
    ):
        super().__init__()
        self.embedding=nn.Embedding(
            vocab_size,
            d_model
        )
        layers=[]
        for _ in range(n_layers):
            layers.append(
                nn.Linear(
                    d_model,
                    d_model,
                )
            )
            layers.append(
                nn.GELU()
            )
        self.layers=nn.Sequential(
            *layers
        )
        self.output=nn.Linear(d_model,vocab_size)
    def forward(self,input_ids):
        x=self.embedding(input_ids)
        x=self.layers(x)
        return self.output(x)

def model_memory(model):
    total=0
    for p in model.parameters():
        total+=(
            p.name()*p.element_size()
        )
    return total

def run(dtype):
    cleanup()
    print("\n"+"="*60)
    print("DType:",dtype)
    print("="*60)
    model=SmallModel()
    model=model.to(device=DEVICE,dtype=dtype)
    model.eval()
    print(
        "Parameter memory:",
        format_bytes(
            model_memory(model)
        )
    )
    print_memory("MODEL LOADED")
    input_ids=torch.randint(
        0,10_000,
        (1,512),
        device=DEVICE
    )
    with torch.inference_mode():
        output=model(input_ids)
    print_memory("AFTER INFERENCE")
    del output
    del input_ids
    del model
    cleanup()
    print_memory("AFTER CLEANUP")
    
def main():
    run(torch.float32)
    run(torch.float16)
    run(torch.bfloat16)
    
if __name__=="__main__":
    main()