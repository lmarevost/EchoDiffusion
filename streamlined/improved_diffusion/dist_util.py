"""
Helpers for distributed training.
"""

import io
import os
import socket

import blobfile as bf
import mpi4py # ADDED 
from mpi4py import MPI
import torch as th
import torch.distributed as dist

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 4

SETUP_RETRY_COUNT = 3


def setup_dist():
    """
    Setup a distributed process group.
    """
    # Checking if the default process group has been initialized
    if dist.is_initialized():
        return
    # with MPI all the processes are grouped in a communicator
    # grouping processes together, allowing them to communicate
    # COMM_WORLD groups all processes together
    comm = MPI.COMM_WORLD
    # choose backend (gloo, mpi or nccl)
    # Use the NCCL backend for distributed GPU training
    # Use the Gloo backend for distributed CPU training.
    # https://pytorch.org/docs/stable/distributed.html
    backend = "gloo" if not th.cuda.is_available() else "nccl"
    # set hostname
    if backend == "gloo":
        hostname = "localhost"
    else:
        # gethostbyname = returns the IP address of the host
        # getfqdn = returns the fully qualified domain name of a host
        hostname = socket.gethostbyname(socket.getfqdn())
    
    # os.environ object returns a dict with user’s environmental variables. 
    os.environ["MASTER_ADDR"] = comm.bcast(hostname, root=0)
    os.environ["RANK"] = str(comm.rank) # ID of each unique process/GPU in the MPI communicator
    os.environ["WORLD_SIZE"] = str(comm.size) # total number of processes/GPU in the MPI comm
    # assign data to root master node (rank 0) to be broadcast to other nodes
    port = comm.bcast(_find_free_port(), root=0)
    os.environ["MASTER_PORT"] = str(port)
    # Initializes the default distributed process group, 
    # and this will also initialize the distributed package
    dist.init_process_group(backend=backend, init_method="env://")


def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        #print('cuda is available')
        #print('return: get rank % gpus per node:',MPI.COMM_WORLD.Get_rank() % GPUS_PER_NODE)
        return th.device(f"cuda:{MPI.COMM_WORLD.Get_rank() % GPUS_PER_NODE}")
    print('returns cpu')
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    """
    # read image data if root node
    if MPI.COMM_WORLD.Get_rank() == 0:
        with bf.BlobFile(path, "rb") as f:
            data = f.read()
    # else data is None
    else:
        data = None
    # broadcast data
    data = MPI.COMM_WORLD.bcast(data)
    
    return th.load(io.BytesIO(data), **kwargs)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
