import torch

def get_device(device=None):
    if device is not None:
        return device
    
    return 'cuda' if torch.cuda.is_available() else 'cpu'

# import sys
# def _debug_sizeof_fmt(num, suffix='B'):
#     ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
#     for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
#         if abs(num) < 1024.0:
#             return "%3.1f %s%s" % (num, unit, suffix)
#         num /= 1024.0
#     return "%.1f %s%s" % (num, 'Yi', suffix)

# for name, size in sorted(((name, sys.getsizeof(value)) for name, value in list(
#                         locals().items())), key= lambda x: -x[1])[:10]:
#     print("{:>30}: {:>8}".format(name, _debug_sizeof_fmt(size)))

# torch.profiler.profile(
#     activities=[torch.profiler.ProfilerActivity.CPU, 
#                 torch.profiler.ProfilerActivity.CUDA],
#     record_shapes=True,
#     profile_memory=True,
#     with_stack=True
# )


# def detailed_memory_report():
#     print("\n=== GPU Memory Report ===")
#     print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
#     print(f"Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
#     print(f"Max allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    
#     # Get memory summary
#     print(torch.cuda.memory_summary())

# def track_tensor_memory():
#     """Track large tensors in memory"""
#     import gc
#     total_params = 0
#     total_grads = 0
    
#     for obj in gc.get_objects():
#         if torch.is_tensor(obj) and obj.is_cuda:
#             size = obj.numel() * obj.element_size()
#             if size > 1024**2:  # Only show tensors > 1MB
#                 print(f"Tensor: {obj.shape}, {size/1024**2:.1f}MB, dtype: {obj.dtype}")
            
#             if hasattr(obj, 'grad') and obj.grad is not None:
#                 total_grads += obj.grad.numel() * obj.grad.element_size()
#             total_params += size
    
#     print(f"Total parameter memory: {total_params/1024**3:.2f}GB")
#     print(f"Total gradient memory: {total_grads/1024**3:.2f}GB")

# def analyze_model_memory(model, optimizer):
#     """Analyze memory usage of model and optimizer"""
    
#     # Model parameters
#     model_params = sum(p.numel() * p.element_size() for p in model.parameters())
#     print(f"Model parameters: {model_params/1024**3:.2f}GB")
    
#     # Model gradients
#     model_grads = sum(p.grad.numel() * p.grad.element_size() 
#                       for p in model.parameters() if p.grad is not None)
#     print(f"Model gradients: {model_grads/1024**3:.2f}GB")
    
#     # Optimizer state (Adam stores momentum and squared gradients)
#     optimizer_memory = 0
#     for group in optimizer.param_groups:
#         for p in group['params']:
#             if p.grad is not None:
#                 state = optimizer.state[p]
#                 for key, value in state.items():
#                     if torch.is_tensor(value):
#                         optimizer_memory += value.numel() * value.element_size()
    
#     print(f"Optimizer state: {optimizer_memory/1024**3:.2f}GB")
#     print(f"Total (model + gradients + optimizer): {(model_params + model_grads + optimizer_memory)/1024**3:.2f}GB")
