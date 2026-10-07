import torch
import torch.nn as nn
from memory_utils import(
    cleanup,
    format_bytes,
    print_memory
)

DEVICE="mps"
class SmallTransformer(nn.Module):
    def __init__(self,
                 vocab_size=10_000,
                 d_model=512,
                 n_heads=8,
                 n_layers=6,
                 d_ff=2048):
        super().__init__()
        self.embedding=nn.Embedding(
            vocab_size,
            d_model
        )
        encoder_layer=nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            batch_first=True
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

def main():
    cleanup()
    print_memory("START")
    model=SmallTransformer().to(DEVICE)
    print_memory("AFTER MODEL LOAD")
    
    batch_size=1
    sequence_length=128
    input_ids=torch.randint(
        0,10_000,(batch_size,sequence_length),
        device=DEVICE
    )
    print_memory("\nInference with autograd....")
    output=model(input_ids)
    print("Output shape:",
          output.shape)
    print_memory("AFTER INFERENCE WITH AUTOGRAD")
    del output

    