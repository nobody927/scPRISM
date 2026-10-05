"""
Distributed training utilities (Real Distributed Utils)
Supports DDP (Distributed Data Parallel)
"""
import os
import torch as th
import torch.distributed as dist


def setup_dist():
    """
    Setup a distributed process group.
    """
    if dist.is_initialized():
        return

    # Check if running under torchrun
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # Multi-GPU mode (DDP)
        th.backends.cudnn.benchmark = True  # accelerate convolutions
        # 1. Set GPU for current process
        local_rank = int(os.environ["LOCAL_RANK"])
        th.cuda.set_device(local_rank)

        # 2. Initialize process group (NCCL is fastest backend for NVIDIA GPUs)
        dist.init_process_group(backend="nccl", init_method="env://")
        print(f"[Init] Process group initialized: rank {get_rank()} / {get_world_size()}")
    else:
        # Single-GPU mode / debug mode
        print("[Init] Running in single-device mode (no DDP).")
        if th.cuda.is_available():
            th.cuda.set_device(0)


def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        if dist.is_initialized():
            # In DDP mode，device should be the GPU bound to current process
            return th.device(f"cuda:{th.cuda.current_device()}")
        return th.device("cuda")
    return th.device("cpu")


def load_state_dict(path, map_location=None):
    """
    Load a state dict from the given path.
    """
    if map_location is None:
        map_location = dev()
    return th.load(path, map_location=map_location)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    Ensure identical initial weights across all GPUs.
    """
    if not dist.is_initialized():
        return

    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)


def get_world_size():
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def barrier():
    """
    Sync barrier: all processes must reach here before continuing.
    """
    if dist.is_initialized():
        dist.barrier()