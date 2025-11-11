# Torch Multi-GPU statevector support
Minimized extension of [`torchquantum`](https://github.com/mit-han-lab/torchquantum) (henceforth `tq`) to allow multi-GPU distributed statevector using `DTensor` from `torch.distributed`. Basic structure is inspired by `tq` but this is a full reimplementation that does _not_ require `tq` as a dependency.

`tqd` provides:
  - `DistributedQuantumDevice`, similar to `QuantumDevice`, but allowing statevector to be distributed across multiple GPUS
  - Gates similar to those in `tq.functional` that operate on `DistributedQuantumDevice` (e.g. `x`, `cy`, `rz`)
  - Modules defining gates and containing trainable parameters similar to those in `tq.operator` (e.g. `X`, `CY`, `RZ`)
  - Measurement of all qubits in Pauli Z (computational) basis
  - Ability to extend the library with your own custom gates (n.b. does NOT check for unitarity!)
  - (2025-05-22) Invertible backpropagation requires `nightly` install of pytorch:
    - For NVIDIA GPU: `pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128`
    - For AMD GPU: `pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/rocm6.4`

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

exact = tqd.measure_allZ(qdev)
```

## Set up
Some rough instructions for different environments. Consult Google if you get stuck.

### GCP
In a GCP VM n-standard-4 with 2x T4 GPUs, follow instructions to set up CUDA drivers: https://cloud.google.com/compute/docs/gpus/install-drivers-gpu#linux

### Installation
So far only tested with `python==3.9`. From this directory:
```bash
pip install .
```

## Quick test
On GCP:
`torchrun --nproc-per-node=2 test_dqd.py`

On Frontier:
(assumes you have a conda environment in `~/.conda/envs/tqd` with this package `pip` installed)
`sbatch --export=NONE batch_test_tqd.sl`

## Development
Currently, it is assumed that gates have either 0 or 1 parameter.

To add custom gates without modifying the library, use the `tqd.custom.register_gate` functionality. They will show up in the `tqd.custom` module.

To further extend the gate set, simply create a new entry in `tqd.matrices.GATE_MAT_DICT`. Functionals and Operators automatically get created from this dictionary.

The `InvertibleUnitary` fundamental module contains basic functionality for depolarizing noise modeling.
The `measure_allZ` measurement function also contains basic functionality to perform postselection at the time of measurement.

## Citation
```
@inproceedings{knitter2025tqd,
      title = {TorchQuantumDistributed},
      author = {Knitter, Oliver and Mei, Jonathan and Yamada, Masako and Roetteler, Martin},
      booktitle = {NeurIPS Workshop on AI for Science: The Reach and Limits of AI for Scientific Discovery},
      year = {2025}
}
```
