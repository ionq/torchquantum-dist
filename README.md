# Torch Multi-GPU statevector support
Extension of `torchquantum` (henceforth `tq`) to allow multi-GPU distributed statevector using `DTensor` from `torch.distributed`

`tqd` provides:
  - `DistributedQuantumDevice`, similar to `QuantumDevice`, but allowing statevector to be distributed across multiple GPUS
  - Gates similar to those in `tq.functional` that operate on `DistributedQuantumDevice` (e.g. `x`, `cy`, `rz`)
  - Modules defining gates and containing trainable parameters similar to those in `tq.operator` (e.g. `X`, `CY`, `RZ`)

## Example usage
```python
import torch

import tqd

nq = 6  # number of qubits
qdev = tqd.DistributedQuantumDevice(nq)

# functional on the qdev
tqd.z(qdev)

# create a stateful RY gate module that tracks its own parameters
ry = tqd.RY(wires=[0], params=torch.pi/3)
ry(qdev)

# operate directly using qdev's own methods
qdev.cx(wires=[0,1])
```

## Set up
Some rough instructions for different environments. Consult Google if you get stuck.

### GCP
In a GCP VM n-standard-4 with 2x T4 GPUs, follow instructions to set up CUDA drivers: https://cloud.google.com/compute/docs/gpus/install-drivers-gpu#linux

### Conda
`conda env create -yf env.yaml`


## Quick test

`torchrun --nproc-per-node=2 test_dqd.py`

## Development
Currently, it is assumed that gates have either 0 or 1 parameter.
To further extend, simply create a new function and add it to `tqd.functional` and append to the list `tqd.FUNC_NAMES`. Operators automatically get created from functionals.