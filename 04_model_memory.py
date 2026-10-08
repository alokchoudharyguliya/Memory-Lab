import torch
import torch.nn as nn
from memory_utils import (
    cleanup,
    format_bytes, 
    print_memory
)
DEVICE="mps"

# Transformers -> Weights, Activations, Temporary tensors => Runtime memory
# parameter memory=sum(parameter.numel*parameter.element_size)
# with FP32 so, 100M * 4 bytes = 400MB
# with FP16 so, 100M * 2 bytes = 200MB
# This is only the weights, the actual inference process needs some memory


class SmallTransformer(nn.Module):
    def __init__(
        self,
        vocab_size=10_000,
        d_model=512,
        n_heads=8,
        n_layers=6,
        d_ff=2048,
    ):
        super().__init__()
        self.embedding=nn.Embedding(
            vocab_size,
            d_model,
        )
        encoder_layer=nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            batch_first=True,
        )
        self.encoder=nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )
        self.output=nn.Linear(
            d_model,
            vocab_size
        )
    def forward(self,input_ids):
        x=self.embedding(input_ids)
        x=self.encoder(x)
        logits=self.output(x)
        return logits

def parameter_memory(model):
    total=0
    for param in model.parameters():
        total+=(
            param.numel()*param.element_size()
        )
    return total

def buffer_memory(model):
    total=0
    for buff in model.buffers():
        total+=(
            buff.numel()*buff.element_size()
        )
    return total
def main():
    cleanup()
    print_memory("START")
    print("\nCreating Model")
    model=SmallTransformer()
    print("\nParameter count",sum(p.numel() for p in model.parameters()))
    print("Parameter memory:",
          format_bytes(
              parameter_memory(model)
          ))
    print("Buffer memory:",
          format_bytes(
              buffer_memory(model)
          ))
    
    print_memory("AFTER CPU MODEL CREATION")
    print("\n Moving model to MPS")
    model=model.to(DEVICE)
    print_memory("AFTER MODEL -> MPS")
    print("\nParameter breakdown:")
    for name, param in model.named_parameters():
        memory=(
            param.numel()*param.element_size()
        )
        print(
            f"{name:60}"
            f"{param.numel():12,} params"
            f"{format_bytes(memory)}"
        )
    del model
    cleanup()
    print_memory("AFTER MODEL DELETION")
    
if __name__=="__main__":
    main()