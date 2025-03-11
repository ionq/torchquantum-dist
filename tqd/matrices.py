import torch

# assumes no batches of params
def rx_mat(params: torch.Tensor) -> torch.Tensor:
    theta = params.type(torch.complex64)
    co = torch.cos(theta / 2)
    jsi = 1j * torch.sin(-theta / 2)
    return torch.stack(
        [torch.cat([co, jsi], dim=-1), torch.cat([jsi, co], dim=-1)], dim=-2
    )

def ry_mat(params: torch.Tensor) -> torch.Tensor:
    theta = params.type(torch.complex64)
    co = torch.cos(theta / 2)
    si = torch.sin(theta / 2)
    return torch.stack(
        [torch.cat([co, -si], dim=-1), torch.cat([si, co], dim=-1)], dim=-2
    )

def rz_mat(params: torch.Tensor) -> torch.Tensor:
    theta = params.type(torch.complex64)
    exp = torch.exp(-0.5j * theta)
    return torch.stack([
        torch.cat([exp, torch.zeros_like(exp)], dim=-1),
        torch.cat([torch.zeros_like(exp), torch.conj(exp)], dim=-1)
    ], dim=-2)

GATE_MAT_DICT = {
    'x': torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64),
    'y': torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64),
    'z': torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64),
    'cx': torch.tensor(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=torch.complex64
    ),
    'cy': torch.tensor(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]], dtype=torch.complex64
    ),
    'cz': torch.tensor(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]], dtype=torch.complex64
    ),
    'rx': rx_mat,
    'ry': ry_mat,
    'rz': rz_mat,
}