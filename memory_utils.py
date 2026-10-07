import gc
import os
import time
import psutil
import torch
PROCESS=psutil.Process(os.getpid())

def format_bytes(num_bytes:int)->int:
    units=["B","KB","MB","GB","TB"]
    value=float(num_bytes)
    for unit in units:
        if value<1024:
            return f"{value:.2f} {unit}"
        value/=1024
    return f"{value:.2f} PB"


def cpu_memory()->int:
    """
    RSS=Resident Set Size
    Roughly, how much physical memory is currently resident for this Python process
    """
    return PROCESS.memory_info().rss

def mps_memory():
    """
    Return MPS memory statistics when available
    """
    if not torch.backends.mps.is_available():
        return None
    allocated=torch.mps.current_allocated_memory()
    driver=torch.mps.driver_allocated_memory()
    return {"allocated":allocated,
            "driver_allocated":driver,}
    
def print_memory(label:str):
    print(f"\n----{label}----")
    
    print(
        f"Process RSS:"
        f"{format_bytes(cpu_memory())}"
    )
    stats=mps_memory()
    
    if stats is not None:
        print(
            f"MPS allocated:",
            f"{format_bytes(stats["allocated"])}"
        )
        print(
            f"MPS driver allocated:"
            f"{format_bytes(stats["driver_allocated"])}"
        )
    
def cleanup():
    """

    
    """
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    time.sleep(0.1)
    
    
def tensor_memory(tensor:torch.Tensor)->int:
    return tensor.numel()*tensor.element_size()

def describe_tensor(name:str, tensor:torch.Tensor):
    print(f"\n{name}")
    print(f"Shpae:          {tuple(tensor.shape)}")
    print(f"dtype:          {tensor.dtype}")
    print(f"device:         {tensor.device}")
    print(f"elements:       {tensor.numel():,}")
    print(f"bytes/element:  {tensor.element_size()}")
    
    print(
        f"storage:          "
        f"{format_bytes(tensor_memory(tensor))}"
    )
    