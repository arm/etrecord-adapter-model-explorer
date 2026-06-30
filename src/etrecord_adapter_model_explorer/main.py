# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Dict

from model_explorer import Adapter, AdapterMetadata, ModelExplorerGraphs

from .builder.builder import ETRecordGraphBuilder
from .constants import ADAPTER_ID, ADAPTER_NAME


class ETRecordAdapter(Adapter):
    metadata = AdapterMetadata(
        id=ADAPTER_ID,
        name=ADAPTER_NAME,
        description="ETRecord adapter for Model Explorer",
        fileExts=["etrecord"],
    )

    def __init__(self):
        super().__init__()

    def convert(self, model_path: str, settings: Dict) -> ModelExplorerGraphs:
        """Convert a given model to a model-explorer compatible format."""

        return {
            "graphCollections": ETRecordGraphBuilder(
                Path(model_path), settings
            ).build()
        }
