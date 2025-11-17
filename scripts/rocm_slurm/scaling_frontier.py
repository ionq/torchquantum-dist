import argparse
import os
import time

import torch

import tqd
import tqd.module


def scaling(batch, nq, world_sz):
    rank = os.environ["RANK"]

    qdev = tqd.DistributedQuantumDevice(
        nq,
        bsz=batch,
        device="cuda",
        world_sz=world_sz,
        invertible=True,
    )

    func_list = [{"func": "ry", "wires": [i], "input_idx": [i]} for i in range(nq)]
    enc = tqd.GeneralEncoder(func_list)
    base_mod = [enc]
    for _ in range(2):
        base_mod = (
            base_mod
            + [tqd.CX(wires=[i, (i + 1) % nq]) for i in range(nq)]
            + [tqd.RY(wires=[i]) for i in range(nq)]
        )
    mod = tqd.module.InvertibleUnitary(base_mod)
    mod.train()

    def fun(qdev, inp):
        qdev.reset_states()
        mod(qdev, inp)
        meas_approx = tqd.measure_allZ(qdev, shots=0, training=True)
        return meas_approx

    # test backprop:
    x_i = torch.nn.Parameter(torch.rand([batch, nq]) * torch.pi / 3)

    opt = torch.optim.Adam([x_i])
    loss_s = []
    for i in range(5):
        if i == 4:
            st = time.time()
        opt.zero_grad()
        out_i = fun(qdev, x_i)

        loss = out_i.abs().sum()
        loss_s.append(loss.item())
        loss.backward()
        opt.step()
        if i == 4:
            elapsed = time.time() - st
    if rank == "0":
        print(elapsed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("batch", type=int)
    parser.add_argument("nq", type=int)

    parser.add_argument("--master_addr", type=str, required=True)
    parser.add_argument("--master_port", type=str, required=True)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    from mpi4py import MPI

    num_gpus_per_node = torch.cuda.device_count()
    # print ("num_gpus_per_node = " + str(num_gpus_per_node), flush=True)
    args = parse_args()

    comm = MPI.COMM_WORLD
    world_size = comm.Get_size()
    global_rank = rank = comm.Get_rank()
    local_rank = int(rank) % int(
        num_gpus_per_node
    )  # local_rank and device are 0 when using 1 GPU per task
    backend = None
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(global_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["MASTER_ADDR"] = str(args.master_addr)
    os.environ["MASTER_PORT"] = str(args.master_port)
    os.environ["NCCL_SOCKET_IFNAME"] = "hsn0"

    if rank == 0:
        print(args, world_size)
    scaling(args.batch, args.nq, world_size)
