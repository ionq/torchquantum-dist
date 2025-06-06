import argparse
import os

import torch

import tqd
import tqd.module

def scaling(batch, nq, world_sz):
    rank = os.environ['RANK']

    qdev = tqd.DistributedQuantumDevice(
        nq,
        bsz=batch,
        device=f'cuda',
        world_sz=world_sz,
        invertible=True,
    )
    
    func_list = [
        {'func': 'ry', 'wires': [i], 'input_idx': [i]} for i in range(nq)
    ]
    enc = tqd.GeneralEncoder(func_list)
    base_mod = [enc] + \
        [tqd.CX(wires=[i, (i+1) % nq]) for i in range(nq)] + \
        [tqd.RY(wires=[i]) for i in range(nq)]
    mod = tqd.module.InvertibleUnitary(base_mod)
    mod.train()
    
    
    def fun(qdev, inp):
        qdev.reset_states()
        mod(qdev, inp)
        meas_approx = tqd.measure_allZ(qdev, shots=0, training=True)
        return meas_approx

    # test backprop:
    x_i = torch.nn.Parameter(torch.rand([batch, nq], device=f'cuda:{rank}') * torch.pi / 3)

    opt = torch.optim.Adam([x_i])
    loss_s = []
    for i in range(5):

        opt.zero_grad()
        out_i = fun(qdev, x_i)

        loss = out_i.abs().sum()
        loss_s.append(loss.item())
        loss.backward()
        opt.step()
    #print(loss_s)


def parse_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('batch', type=int)
    arg_parser.add_argument('nq', type=int)
    arg_parser.add_argument('world_sz', type=int)

    args = arg_parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    scaling(args.batch, args.nq, args.world_sz)
