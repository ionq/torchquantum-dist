"""Tests for distributed quantum device operations."""

from __future__ import annotations

import torch
from torch.distributed.tensor import DTensor, Partial

import tqd

from .conftest import requires_distributed


@requires_distributed
class TestDistributedQuantumDevice:
    """Tests for the DistributedQuantumDevice class."""

    def test_basic_operations(self) -> None:
        """Test class method, module, functional, registration, and correctness."""
        nq = 3
        world_sz = 2
        wire = 1

        qdev = tqd.DistributedQuantumDevice(nq, device="cuda", world_sz=world_sz)

        # test class method
        qdev.y(wires=[wire])

        # test Module
        cx_gate = tqd.CX(wires=[wire, (wire + 1) % nq])
        cx_gate(qdev)

        # test functional
        tqd.rz(qdev, wires=[wire], params=torch.pi / 3)

        # test registration
        tqd.custom.register_gate("id", torch.eye(2, dtype=torch.cfloat))
        tqd.custom.id(qdev, wires=[1])

        states_tq = torch.Tensor(
            [
                [
                    [[0.0, 0.0], [0.0, 0.0]],
                    [[0.0, 0.0], [-0.5, torch.sqrt(torch.tensor([3])) / 2]],
                ],
                [[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]],
            ]
        )

        assert torch.allclose(states_tq, qdev.states.full_tensor().cpu()[0])

    def test_noiseless_measurement(self) -> None:
        """Test noiseless measurement."""
        nq = 3
        world_sz = 2
        wire = 1

        qdev = tqd.DistributedQuantumDevice(nq, device="cuda", world_sz=world_sz)
        qdev.y(wires=[wire])
        cx_gate = tqd.CX(wires=[wire, (wire + 1) % nq])
        cx_gate(qdev)
        tqd.rz(qdev, wires=[wire], params=torch.pi / 3)

        meas = tqd.measure_allZ(qdev)
        meas_tq = torch.Tensor([[1.0, -1.0, -1.0]])

        assert torch.allclose(meas_tq, meas.cpu())

    def test_noisy_measurement_sampling(self) -> None:
        """Test noisy non-differentiable (sampling) measurement."""
        nq = 3
        world_sz = 2
        wire = 1

        qdev = tqd.DistributedQuantumDevice(nq, device="cuda", world_sz=world_sz)
        qdev.y(wires=[wire])
        cx_gate = tqd.CX(wires=[wire, (wire + 1) % nq])
        cx_gate(qdev)
        tqd.rz(qdev, wires=[wire], params=torch.pi / 3)

        meas_tq = torch.Tensor([[1.0, -1.0, -1.0]])
        meas_samp = tqd.measure_allZ(qdev, shots=1024, training=False)

        assert torch.allclose(meas_tq, meas_samp.cpu())

    def test_noisy_measurement_differentiable(self) -> None:
        """Test noisy differentiable (approximate) measurement."""
        nq = 3
        world_sz = 2
        wire = 1

        qdev = tqd.DistributedQuantumDevice(nq, device="cuda", world_sz=world_sz)
        qdev.y(wires=[wire])
        cx_gate = tqd.CX(wires=[wire, (wire + 1) % nq])
        cx_gate(qdev)
        tqd.rz(qdev, wires=[wire], params=torch.pi / 3)

        meas_tq = torch.Tensor([[1.0, -1.0, -1.0]])
        meas_approx = tqd.measure_allZ(qdev, shots=1024, training=True)

        assert torch.allclose(meas_tq, meas_approx.cpu())


@requires_distributed
class TestNoisyMeasurement:
    """Tests for noisy measurement functionality."""

    def test_noisy_measurement_accuracy(self) -> None:
        """Test noisy measurement accuracy with many shots."""
        nq = 3
        world_sz = 2
        wire = 1

        qdev = tqd.DistributedQuantumDevice(nq, device="cuda", world_sz=world_sz)
        qdev.rx(wires=[wire], params=torch.pi / 6)

        meas_tq = torch.Tensor([[1.0, torch.sqrt(torch.tensor([3])) / 2, 1.0]])

        # test noisy non-differentiable (sampling) measurement
        meas_samp, _ = tqd.measure_allZ(
            qdev, shots=int(1e5), training=False, postselect_cond={2: 0}
        )

        # test noisy differentiable (approximate) measurement
        meas_approx = tqd.measure_allZ(qdev, shots=int(1e5), training=True)

        assert torch.allclose(meas_tq, meas_approx.cpu(), rtol=1e-3, atol=1e-2)
        assert torch.allclose(meas_tq, meas_samp.cpu(), rtol=1e-3, atol=1e-2)


@requires_distributed
class TestEncoder:
    """Tests for the GeneralEncoder."""

    def test_general_encoder(self) -> None:
        """Test encoder functionality."""
        nq = 3
        world_sz = 2

        qdev = tqd.DistributedQuantumDevice(nq, device="cuda", world_sz=world_sz)

        func_list = [
            {"func": "rx", "wires": [0], "input_idx": [0]},
            {"func": "ry", "wires": [1], "input_idx": [1]},
            {"func": "rz", "wires": [2], "input_idx": [2]},
        ]
        x = torch.Tensor([[1, 2, 3]])

        enc = tqd.GeneralEncoder(func_list)
        enc(qdev, x)
        states = qdev.states

        states_tq = torch.Tensor(
            [
                [
                    [
                        [[0.03354074, -0.47297207], [0.00000000, 0.00000000]],
                        [[0.05223661, -0.73661041], [0.00000000, 0.00000000]],
                    ],
                    [
                        [[-0.25838584, -0.01832339], [0.00000000, 0.00000000]],
                        [[-0.40241212, -0.02853699], [0.00000000, 0.00000000]],
                    ],
                ]
            ]
        )

        assert torch.allclose(states_tq, states.full_tensor().cpu())


@requires_distributed
class TestGradients:
    """Tests for gradient computation."""

    def test_gradient_computation(self) -> None:
        """Test gradient computation through quantum operations."""
        nq = 3
        world_sz = 2

        qdev = tqd.DistributedQuantumDevice(nq, device="cuda", world_sz=world_sz)
        p = torch.nn.Parameter(
            torch.Tensor([torch.pi / 3, -torch.pi / 3, torch.pi / 6])
        )
        [qdev.ry(wires=[i], params=p[i]) for i in range(nq)]
        [qdev.cx(wires=[i, (i + 1) % nq]) for i in range(nq)]

        qdev.states.abs().sum().backward()
        grad_dist = DTensor.from_local(
            p.grad[None], qdev.device_mesh, placements=[Partial()]
        )
        grad = grad_dist.full_tensor()

        assert torch.allclose(
            grad.cpu(),
            torch.Tensor([0.3061861991882324, -0.3061861991882324, 0.65973961353302]),
        )


@requires_distributed
class TestQubitGroupings:
    """Tests for qubit groupings inside dtensor."""

    def test_groupings(self) -> None:
        """Test qubit groupings inside dtensor."""
        max_dtensor_dims = 7
        nq = 7
        wire = 1
        world_sz = 2

        qdev = tqd.DistributedQuantumDevice(
            nq, device="cuda", world_sz=world_sz, max_dtensor_dims=max_dtensor_dims
        )

        # test class method
        qdev.y(wires=[wire])

        # test Module
        cx_gate = tqd.CX(wires=[wire, (wire + 3) % nq])
        cx_gate(qdev)

        # test functional
        tqd.rz(qdev, wires=[(wire + 2) % nq], params=torch.pi / 3)

        # test registration
        tqd.custom.register_gate("id", torch.eye(2, dtype=torch.cfloat))
        tqd.custom.id(qdev, wires=[1])

        qdev.canonicalize()
        states_tq = torch.Tensor(
            [
                [
                    [
                        [
                            [
                                [
                                    [
                                        [
                                            [[0.0000, 0.0000], [0.0000, 0.0000]],
                                            [[0.0000, 0.0000], [0.0000, 0.0000]],
                                        ],
                                        [
                                            [[0.0000, 0.0000], [0.0000, 0.0000]],
                                            [[0.0000, 0.0000], [0.0000, 0.0000]],
                                        ],
                                    ],
                                    [
                                        [
                                            [[0.0000, 0.0000], [0.0000, 0.0000]],
                                            [[0.0000, 0.0000], [0.0000, 0.0000]],
                                        ],
                                        [
                                            [[0.0000, 0.0000], [0.0000, 0.0000]],
                                            [[0.0000, 0.0000], [0.0000, 0.0000]],
                                        ],
                                    ],
                                ],
                                [
                                    [
                                        [
                                            [[0.0000, 0.0000], [0.0000, 0.0000]],
                                            [[0.0000, 0.0000], [0.0000, 0.0000]],
                                        ],
                                        [
                                            [[0.0000, 0.0000], [0.0000, 0.0000]],
                                            [[0.0000, 0.0000], [0.0000, 0.0000]],
                                        ],
                                    ],
                                    [
                                        [
                                            [[0.0000, 0.0000], [0.0000, 0.0000]],
                                            [[0.0000, 0.0000], [0.0000, 0.0000]],
                                        ],
                                        [
                                            [[0.0000, 0.0000], [0.0000, 0.0000]],
                                            [[0.0000, 0.0000], [0.0000, 0.0000]],
                                        ],
                                    ],
                                ],
                            ],
                            [
                                [
                                    [
                                        [
                                            [[0.0000, 0.0000], [0.0000, 0.0000]],
                                            [[0.0000, 0.0000], [0.0000, 0.0000]],
                                        ],
                                        [
                                            [
                                                [
                                                    0.5000,
                                                    torch.sqrt(torch.Tensor([3])) / 2,
                                                ],
                                                [0.0000, 0.0000],
                                            ],
                                            [[0.0000, 0.0000], [0.0000, 0.0000]],
                                        ],
                                    ],
                                    [
                                        [
                                            [[0.0000, 0.0000], [0.0000, 0.0000]],
                                            [[0.0000, 0.0000], [0.0000, 0.0000]],
                                        ],
                                        [
                                            [[0.0000, 0.0000], [0.0000, 0.0000]],
                                            [[0.0000, 0.0000], [0.0000, 0.0000]],
                                        ],
                                    ],
                                ],
                                [
                                    [
                                        [
                                            [[0.0000, 0.0000], [0.0000, 0.0000]],
                                            [[0.0000, 0.0000], [0.0000, 0.0000]],
                                        ],
                                        [
                                            [[0.0000, 0.0000], [0.0000, 0.0000]],
                                            [[0.0000, 0.0000], [0.0000, 0.0000]],
                                        ],
                                    ],
                                    [
                                        [
                                            [[0.0000, 0.0000], [0.0000, 0.0000]],
                                            [[0.0000, 0.0000], [0.0000, 0.0000]],
                                        ],
                                        [
                                            [[0.0000, 0.0000], [0.0000, 0.0000]],
                                            [[0.0000, 0.0000], [0.0000, 0.0000]],
                                        ],
                                    ],
                                ],
                            ],
                        ],
                        [
                            [
                                [
                                    [
                                        [
                                            [[0.0000, 0.0000], [0.0000, 0.0000]],
                                            [[0.0000, 0.0000], [0.0000, 0.0000]],
                                        ],
                                        [
                                            [[0.0000, 0.0000], [0.0000, 0.0000]],
                                            [[0.0000, 0.0000], [0.0000, 0.0000]],
                                        ],
                                    ],
                                    [
                                        [
                                            [[0.0000, 0.0000], [0.0000, 0.0000]],
                                            [[0.0000, 0.0000], [0.0000, 0.0000]],
                                        ],
                                        [
                                            [[0.0000, 0.0000], [0.0000, 0.0000]],
                                            [[0.0000, 0.0000], [0.0000, 0.0000]],
                                        ],
                                    ],
                                ],
                                [
                                    [
                                        [
                                            [[0.0000, 0.0000], [0.0000, 0.0000]],
                                            [[0.0000, 0.0000], [0.0000, 0.0000]],
                                        ],
                                        [
                                            [[0.0000, 0.0000], [0.0000, 0.0000]],
                                            [[0.0000, 0.0000], [0.0000, 0.0000]],
                                        ],
                                    ],
                                    [
                                        [
                                            [[0.0000, 0.0000], [0.0000, 0.0000]],
                                            [[0.0000, 0.0000], [0.0000, 0.0000]],
                                        ],
                                        [
                                            [[0.0000, 0.0000], [0.0000, 0.0000]],
                                            [[0.0000, 0.0000], [0.0000, 0.0000]],
                                        ],
                                    ],
                                ],
                            ],
                            [
                                [
                                    [
                                        [
                                            [[0.0000, 0.0000], [0.0000, 0.0000]],
                                            [[0.0000, 0.0000], [0.0000, 0.0000]],
                                        ],
                                        [
                                            [[0.0000, 0.0000], [0.0000, 0.0000]],
                                            [[0.0000, 0.0000], [0.0000, 0.0000]],
                                        ],
                                    ],
                                    [
                                        [
                                            [[0.0000, 0.0000], [0.0000, 0.0000]],
                                            [[0.0000, 0.0000], [0.0000, 0.0000]],
                                        ],
                                        [
                                            [[0.0000, 0.0000], [0.0000, 0.0000]],
                                            [[0.0000, 0.0000], [0.0000, 0.0000]],
                                        ],
                                    ],
                                ],
                                [
                                    [
                                        [
                                            [[0.0000, 0.0000], [0.0000, 0.0000]],
                                            [[0.0000, 0.0000], [0.0000, 0.0000]],
                                        ],
                                        [
                                            [[0.0000, 0.0000], [0.0000, 0.0000]],
                                            [[0.0000, 0.0000], [0.0000, 0.0000]],
                                        ],
                                    ],
                                    [
                                        [
                                            [[0.0000, 0.0000], [0.0000, 0.0000]],
                                            [[0.0000, 0.0000], [0.0000, 0.0000]],
                                        ],
                                        [
                                            [[0.0000, 0.0000], [0.0000, 0.0000]],
                                            [[0.0000, 0.0000], [0.0000, 0.0000]],
                                        ],
                                    ],
                                ],
                            ],
                        ],
                    ]
                ]
            ]
        )

        assert torch.allclose(
            states_tq,
            qdev.states.full_tensor().cpu()[0].reshape((1,) + (2,) * nq + (2,)),
        )

        # test noisy non-differentiable (sampling) measurement
        _ = tqd.measure_allZ(qdev, shots=int(1e5), training=False)

        # test noisy differentiable (approximate) measurement
        _ = tqd.measure_allZ(qdev, shots=int(1e5), training=True)
