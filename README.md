tensor storage VS process RSS, MPS allocated, MPS driver allocated

tensor memory=number of elements X bytes per element

mps tensor memory + cpu tensor memory
data movement 


allocate tensor - temporary buffers and activations


Mac / MPS
   │
   │ understand memory behavior
   ▼
Colab / NVIDIA
   │
   ├── VRAM
   ├── pinned host memory
   ├── H2D / D2H
   ├── cudaMemcpyAsync
   ├── CUDA streams
   ├── CUDA events
   └── allocator / memory pools