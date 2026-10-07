<!---
SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
SPDX-License-Identifier: Apache-2.0
--->

# ETRecord Adapter for Model Explorer

`etrecord-adapter-model-explorer` is the standalone Model Explorer adapter
for ExecuTorch ETRecord files using the `.etrecord` or `.bin` extension.

This repository contains the ETRecord-specific adapter code, tests, and
fixtures. For a combined install that wires the PTE adapter, ETRecord adapter,
and ETDump data provider together, use the umbrella package
`executorch-extension-model-explorer`.

## Installation

Install the standalone adapter package:

```bash
pip install etrecord-adapter-model-explorer
```

The adapter expects an environment with the ExecuTorch Python packages
available.

## Usage

Run Model Explorer with the standalone ETRecord adapter enabled:

```bash
model-explorer --extensions=etrecord_adapter_model_explorer
```

Then open an ETRecord stored as an `.etrecord` or `.bin` file.

## Delegate visibility

Delegated ExecuTorch programs often collapse a backend-supported subgraph into
an opaque `executorch_call_delegate` node. If an ETRecord is generated from the
post-delegation edge graph, this adapter can only display that delegate call for
the delegated region because the original edge operators are no longer present
in the recorded edge graph.

To keep the original operators visible, generate the ETRecord with the
pre-delegation `EdgeProgramManager` as `edge_dialect_program` and the delegated
`ExecutorchProgramManager` as `executorch_program`:

```python
import copy
import torch
from executorch.backends.arm import EthosUCompileSpec, EthosUPartitioner
from executorch.devtools import generate_etrecord
from executorch.exir import to_edge, to_edge_transform_and_lower

exported_program = torch.export.export(model.eval(), example_inputs)

edge_program = to_edge(exported_program)
edge_program_for_record = copy.deepcopy(edge_program)

compile_spec = EthosUCompileSpec("ethos-u55-128")
delegated_edge_program = to_edge_transform_and_lower(
    programs=exported_program,
    partitioner=[EthosUPartitioner(compile_spec)],
)
executorch_program = delegated_edge_program.to_executorch()

generate_etrecord(
    "model.etrecord",
    edge_dialect_program=edge_program_for_record,
    executorch_program=executorch_program,
)
```

Backends that emit delegate debug mapping can also let the adapter group those
operators under a delegate namespace such as
`delegates/CoreMLBackend/partition_0`. Backends that do not emit that mapping
can still show the pre-delegation operators, but delegate grouping may be
limited.

## Development

Install the development dependencies with pip 25.1 or newer:

```bash
python -m pip install -e . --group dev \
  --find-links https://download.pytorch.org/whl/cpu/torch/
```

The dev group selects CPU-only PyTorch wheels on Linux and Windows, and the
regular macOS wheel. The `--find-links` option supplies PyTorch's wheel
location alongside your configured package index.

Local ETRecord test fixtures live under
`src/etrecord_adapter_model_explorer/tests/fixtures/`.

```bash
pytest
```
