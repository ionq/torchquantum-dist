"""Run tqd tests on Frontier using MPI.

This script sets up the distributed environment using MPI and runs
the test suite. For standard pytest usage, use `torchrun` with the
tests in the tests/ directory instead.

Usage:
    srun python test_tqd_frontier.py --master_addr <addr> --master_port <port>
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


def main() -> None:
    """Set up MPI environment and run pytest."""
    from mpi4py import MPI

    import torch

    parser = argparse.ArgumentParser(description="Run tqd tests on Frontier with MPI")
    parser.add_argument("--master_addr", type=str, required=True)
    parser.add_argument("--master_port", type=str, required=True)

    args = parser.parse_args()

    num_gpus_per_node = torch.cuda.device_count()
    print(f"num_gpus_per_node = {num_gpus_per_node}", flush=True)

    comm = MPI.COMM_WORLD
    world_size = comm.Get_size()
    global_rank = rank = comm.Get_rank()
    local_rank = int(rank) % int(num_gpus_per_node)

    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(global_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["MASTER_ADDR"] = str(args.master_addr)
    os.environ["MASTER_PORT"] = str(args.master_port)
    os.environ["NCCL_SOCKET_IFNAME"] = "hsn0"

    # Initialize distributed before running tests
    torch.distributed.init_process_group(backend="nccl")

    # Run pytest on the tests directory
    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    tests_dir = os.path.join(repo_root, "tests")

    result = subprocess.run(
        [sys.executable, "-m", "pytest", tests_dir, "-v"],
        cwd=repo_root,
    )

    torch.distributed.destroy_process_group()
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
