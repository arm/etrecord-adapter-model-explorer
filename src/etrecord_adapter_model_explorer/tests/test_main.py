# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

import unittest
from pathlib import Path
from unittest.mock import patch, sentinel

from .. import main as main_module
from ..main import ETRecordAdapter

_ASSERT = unittest.TestCase()


def test_convert_delegates_to_graph_builder():
    settings = {"example": True}

    with patch.object(main_module, "ETRecordGraphBuilder") as mock_builder:
        mock_builder.return_value.build.return_value = (
            sentinel.graph_collections
        )

        graphs = ETRecordAdapter().convert("model.etrecord", settings)

    mock_builder.assert_called_once_with(Path("model.etrecord"), settings)
    mock_builder.return_value.build.assert_called_once_with()
    _ASSERT.assertEqual(
        graphs, {"graphCollections": sentinel.graph_collections}
    )
