# ETRecord Adapter for Model Explorer

`etrecord-adapter-model-explorer` is the standalone Model Explorer adapter
for ExecuTorch `.etrecord` files.

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

Then open an `.etrecord` file.

## Development

Local ETRecord test fixtures live under `tests/fixtures/`.

```bash
pytest
```
