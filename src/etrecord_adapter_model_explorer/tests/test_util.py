# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

import unittest
from typing import cast

from executorch.devtools.debug_format.base_schema import OperatorNode

from etrecord_adapter_model_explorer.builder.util import (
    format_op_label,
    module_namespace,
    node_label,
)

_ASSERT = unittest.TestCase()


def test_node_label_prefers_last_module_type():
    node = OperatorNode(
        name="aten_relu_default_",
        op="aten.relu.default",
        metadata={
            "nn_module_stack": {
                "L__self__": ("", "__main__.SimpleConvNet"),
                "layer": ("features.relu", "torch.nn.modules.activation.ReLU"),
            }
        },
    )

    _ASSERT.assertEqual(node_label(node), "ReLU")


def test_node_label_falls_back_to_op_name_when_only_root_module_is_available():
    node = OperatorNode(
        name="aten_view_copy_default_",
        op="aten.view_copy.default",
        metadata={
            "nn_module_stack": {
                "L__self__": ("", "__main__.SimpleConvNet"),
            }
        },
    )

    _ASSERT.assertEqual(node_label(node), "aten::view_copy")


def test_module_namespace_skips_empty_names_and_preserves_order():
    module_stack = {
        "L__self__": ("", "__main__.SimpleConvNet"),
        "block": ("encoder.block", "pkg.Block"),
        "proj": ("projection", "pkg.Linear"),
    }

    _ASSERT.assertEqual(
        module_namespace(module_stack), "encoder.block/projection"
    )


def test_node_label_falls_back_to_op_name_without_module_stack():
    class DummyNode:
        metadata = {}
        op = "aten.add.Tensor"
        name = "aten_add_tensor_"

    _ASSERT.assertEqual(
        node_label(cast(OperatorNode, DummyNode())), "aten::add"
    )


def test_format_op_label_strips_overload_suffixes():
    _ASSERT.assertEqual(
        format_op_label("aten.view_copy.default"), "aten::view_copy"
    )
    _ASSERT.assertEqual(format_op_label("aten.add.Tensor"), "aten::add")
    _ASSERT.assertEqual(
        format_op_label("aten::sigmoid.default"), "aten::sigmoid"
    )


def test_format_op_label_handles_fallback_inputs():
    # Labels without a namespace are unchanged
    _ASSERT.assertEqual(format_op_label("custom_op"), "custom_op")

    # Missing or malformed labels are handled safely for display
    _ASSERT.assertEqual(format_op_label(None), "")
    _ASSERT.assertEqual(format_op_label("::relu.default"), "::relu.default")
