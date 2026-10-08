import gc
import torch
import torch.nn as nn
from memory_utils import(
    cleanup,
    cpu_memory,
    format_bytes, format_bytes,mps_memory
)
DEVICE="mps"

VOCAB_SIZE=10_000
D_MODEL=512
NUM_HEADS=8
NUM_LAYERS=6
FF_DIM=2048

DTYPE=torch.float16

class MemoryTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding=nn.Embedding(
            VOCAB_SIZE,D_MODEL
        )
        encoder_layer=nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=NUM_HEADS,
            dim_feedforward=FF_DIM,
            batch_first=True
        )
        self.encoder=nn.TransformerEncoder(
            encoder_layer,
            NUM_LAYERS,
        )
        self.output=nn.Linear(D_MODEL,VOCAB_SIZE)
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
def current_mps_memory():
    if not torch.backends.mps.is_available():
        return 0
    return torch.mps.current_allocated_memory()
def measure_stage():
    gc.collect()
    time.sleep(0.05)
    return {
        "rss":cpu_memory(),
        "mps":current_mps_memory()
    }
def print_stage(name, baseline):
    current=measure_stage()
    rss_delta=(current["rss"]-baseline["rss"])
    mps_delta=(current["mps"]-baseline["mps"])
    
    print(
        f"{name:30}"
        f""
    )