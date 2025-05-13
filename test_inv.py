import os

import torch
from torch.distributed.tensor import DTensor, Partial, Replicate
from torch.autograd import gradcheck

import tqd
import tqd.module

def test_inv(verbose=False):
    """
    invertible gradient
    """
    rank = os.environ['RANK']
    nq = 3
    world_sz = 2
    wire = 1

    qdev = tqd.DistributedQuantumDevice(
        nq,
        device=f'cuda',
        world_sz=world_sz,
        invertible=True,
    )

    # test registration
    tqd.custom.register_gate('i', torch.eye(2, dtype=torch.cfloat))
    
    def fun(qdev, inp):
        func_list = [
            {'func': 'ry', 'wires': [0], 'input_idx': [0]},
            {'func': 'ry', 'wires': [1], 'input_idx': [1]},
            {'func': 'ry', 'wires': [2], 'input_idx': [2]},
        ]
        enc = tqd.GeneralEncoder(func_list)
        base_mod = [enc] + [tqd.CX(wires=[i, (i+1) % nq]) for i in range(qdev.n_wires)] + [tqd.custom.I(wires=[1])]
        mod = tqd.module.InvertibleUnitary(base_mod)
        mod.train()
        mod(qdev, inp)
        #meas_approx = tqd.measure_allZ(qdev, shots=int(1e5), training=True)
        return qdev._states
        #return meas_approx

    if verbose:
        print(f'after {rank} {qdev.states}')
        print(f'done {qdev.states.full_tensor()}')

    # test backprop: cast as complex for sake of gradcheck
    x = torch.nn.Parameter(torch.tensor([[torch.pi/3, -torch.pi/3, torch.pi/6]]))

    out = fun(qdev, x)
    out.abs().sum().backward()
    x_grad_dist = DTensor.from_local(x.grad, qdev.device_mesh, placements=[Partial()])
    x_grad = x_grad_dist.redistribute(qdev.device_mesh, placements=[Replicate()])
    assert torch.allclose(x_grad.to_local().cpu(), torch.tensor([[ 0.30618620, -0.30618620,  0.65973961]]))

    if rank == '0':
        print('inverse test passed!')

if __name__ == "__main__":
    test_inv(False)