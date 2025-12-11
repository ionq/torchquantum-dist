"""Tests for invertible gradient computation."""

from __future__ import annotations

import torch
from torch.distributed.tensor import DTensor, Partial

import tqd
import tqd.module

from .conftest import requires_distributed


@requires_distributed
class TestInvertibleGradient:
    """Tests for invertible gradient computation."""

    def test_invertible_vs_standard_gradient(self) -> None:
        """Test that invertible and standard gradients produce correct results."""
        nq = 3
        world_sz = 2

        qdev = tqd.DistributedQuantumDevice(
            nq,
            device="cuda",
            world_sz=world_sz,
            invertible=False,
        )
        qdev_i = tqd.DistributedQuantumDevice(
            nq,
            device="cuda",
            world_sz=world_sz,
            invertible=True,
        )

        # register custom gate
        tqd.custom.register_gate("id", torch.eye(2, dtype=torch.cfloat))

        def run_circuit(qdev: tqd.DistributedQuantumDevice, inp: torch.Tensor):
            func_list = [
                {"func": "ry", "wires": [0], "input_idx": [0]},
                {"func": "ry", "wires": [1], "input_idx": [1]},
                {"func": "ry", "wires": [2], "input_idx": [2]},
            ]
            enc = tqd.GeneralEncoder(func_list)
            base_mod = (
                [enc]
                + [tqd.CX(wires=[i, (i + 1) % nq]) for i in range(qdev.n_wires)]
                + [tqd.custom.ID(wires=[1])]
            )
            if qdev.invertible:
                mod = tqd.module.InvertibleUnitary(base_mod)
                mod.train()
                mod(qdev, inp)
            else:
                enc(qdev, inp)
                for m_ in base_mod[1:]:
                    m_.train()
                    m_(qdev)

            meas_approx = tqd.measure_allZ(qdev, shots=0, training=True)
            return meas_approx

        # Test invertible version
        x_i = torch.nn.Parameter(
            torch.tensor([[torch.pi / 3, -torch.pi / 3, torch.pi / 6]])
        )
        out_i = run_circuit(qdev_i, x_i)
        out_i.abs().sum().backward()
        x_i_grad_dist = DTensor.from_local(
            x_i.grad, qdev_i.device_mesh, placements=[Partial()]
        )
        _ = x_i_grad_dist.full_tensor()  # Verify invertible gradient computes

        # Test standard version
        x = torch.nn.Parameter(
            torch.tensor([[torch.pi / 3, -torch.pi / 3, torch.pi / 6]])
        )
        out = run_circuit(qdev, x)
        out.abs().sum().backward()
        x_grad_dist = DTensor.from_local(
            x.grad, qdev.device_mesh, placements=[Partial()]
        )
        x_grad = x_grad_dist.full_tensor()

        # Verify states
        assert torch.allclose(
            torch.view_as_complex(qdev.states.full_tensor().cpu()),
            torch.tensor(
                [
                    [
                        [
                            [0.7244 + 0.0j, -0.0647 + 0.0j],
                            [-0.1121 + 0.0j, 0.4183 + 0.0j],
                        ],
                        [
                            [-0.2415 + 0.0j, 0.1941 + 0.0j],
                            [0.1121 + 0.0j, -0.4183 + 0.0j],
                        ],
                    ]
                ]
            ),
            atol=1e-3,
            rtol=1e-3,
        )

        # Verify gradients
        assert torch.allclose(
            x_grad.cpu(), torch.tensor([[-0.80801272, 1.55801260, -0.37500000]])
        )
