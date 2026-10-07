# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

import torch
from executorch.devtools.debug_format.base_schema import (
    OperatorNode,
    ValueNode,
)

from etrecord_adapter_model_explorer.builder.metadata import (
    build_input_metadata,
    build_node_attrs,
    build_output_metadata,
)
from etrecord_adapter_model_explorer.tests.helpers import key_value_attrs


def test_build_node_attrs_skips_empty_metadata_values():
    node = OperatorNode(
        name="aten_add_tensor_",
        op="aten.add.Tensor",
        inputs=[
            ValueNode(
                name="weight_",
                output_shapes=[[1, 2]],
                dtype="torch.float32",
            ),
            ValueNode(name="stride_0", val=[1, 1]),
        ],
        named_args=[
            "weight",
            "stride",
            "alpha",
            "beta",
            "gamma",
            "delta",
            "epsilon",
        ],
        output_shapes=[[1, 2]],
        metadata={
            "debug_handle": 9,
            "tag": "math",
            "stack_trace": "skip me",
            "nn_module_stack": {"root": ("", "Model")},
            "skip_none": None,
            "skip_list": [],
            "skip_dict": {},
        },
    )

    attrs = key_value_attrs(build_node_attrs(node))

    assert attrs == {
        "Node Name": "aten_add_tensor_",
        "Op Type": "aten.add.Tensor",
        "weight": "shape=[1, 2], dtype=torch.float32",
        "stride": "[1, 1]",
        "Output Shapes": "[1, 2]",
        "debug_handle": "9",
        "tag": "math",
    }


def test_build_node_attrs_includes_parameter_values_from_state_dict():
    node = OperatorNode(
        name="aten_convolution_default_",
        op="aten.convolution.default",
        inputs=[
            ValueNode(name="p_conv1_weight_", output_shapes=[[1, 1, 3, 3]]),
            ValueNode(name="p_conv1_bias_", output_shapes=[[1]]),
            ValueNode(name="padding_1", val=[1, 1]),
        ],
        named_args=["weight", "bias", "padding"],
        output_shapes=[[1, 1, 8, 8]],
        metadata={"debug_handle": 1},
    )

    attrs = key_value_attrs(
        build_node_attrs(
            node,
            state_dict={
                "conv1.weight": torch.arange(9, dtype=torch.float32).reshape(
                    1, 1, 3, 3
                ),
                "conv1.bias": torch.tensor([0.5], dtype=torch.float32),
            },
            placeholder_to_state_name={
                "p_conv1_weight": "conv1.weight",
                "p_conv1_bias": "conv1.bias",
            },
        )
    )

    assert attrs["weight"].startswith(
        "shape=[1, 1, 3, 3], dtype=torch.float32, values=[0.000000, 1.000000"
    )
    assert attrs["bias"] == (
        "shape=[1], dtype=torch.float32, values=[0.500000]"
    )
    assert attrs["padding"] == "[1, 1]"


def test_build_node_attrs_includes_missing_named_arg_values():
    node = OperatorNode(
        name="aten_mean_dim_",
        op="aten.mean.dim",
        inputs=[
            ValueNode(name="input_", output_shapes=[[1, 4, 8, 8]]),
            ValueNode(name="dim_0", val=[-1, -2]),
        ],
        named_args=["self", "dim", "keepdim", "dtype"],
        output_shapes=[[1, 4, 1, 1]],
        metadata={"debug_handle": 3},
    )

    attrs = key_value_attrs(
        build_node_attrs(
            node,
            named_arg_values={
                "keepdim": True,
                "dtype": None,
            },
        )
    )

    assert attrs["dim"] == "[-1, -2]"
    assert attrs["keepdim"] == "True"
    assert attrs["dtype"] == "None"


def test_build_node_attrs_ignores_omitted_self_and_input_arguments():
    node = OperatorNode(
        name="custom_op",
        op="custom.op",
        inputs=[ValueNode(name="value_", output_shapes=[[1]])],
        named_args=["value", "self", "input", "alpha"],
    )

    attrs = key_value_attrs(
        build_node_attrs(
            node,
            named_arg_values={
                "self": "ignored",
                "input": "ignored",
                "alpha": 1,
            },
        )
    )

    assert "self" not in attrs
    assert "input" not in attrs
    assert attrs["alpha"] == "1"


def test_build_node_attrs_ignores_missing_named_argument_values():
    node = OperatorNode(
        name="custom_op",
        op="custom.op",
        inputs=[ValueNode(name="value_", output_shapes=[[1]])],
        named_args=["value", "provided", "missing"],
    )

    attrs = key_value_attrs(
        build_node_attrs(
            node,
            named_arg_values={"provided": True},
        )
    )

    assert attrs["provided"] == "True"
    assert "missing" not in attrs


def test_build_node_attrs_uses_derived_name_for_numbered_input():
    """Strip generated numeric suffixes from ValueNode attribute names."""
    node = OperatorNode(
        name="custom_op",
        op="custom.op",
        inputs=[
            ValueNode(
                name="value_0",  # _0 will be stripped
                output_shapes=[[1]],
                dtype="torch.float32",
            ),
        ],
        named_args=["other"],  # derived "value" name will take precedence
    )

    attrs = key_value_attrs(build_node_attrs(node))

    assert attrs["value"] == "shape=[1], dtype=torch.float32"


def test_build_node_attrs_formats_tuple_named_argument_as_shape():
    node = OperatorNode(
        name="aten_convolution_default_",
        op="aten.convolution.default",
        inputs=[
            ValueNode(name="input_", output_shapes=[[1, 3, 32, 32]]),
            ValueNode(name="weight_", output_shapes=[[8, 3, 3, 3]]),
            ValueNode(name="bias_", output_shapes=[[8]]),
        ],
        named_args=["input", "weight", "bias", "stride"],
    )

    attrs = key_value_attrs(
        build_node_attrs(
            node,
            named_arg_values={"stride": (1, 1)},
        )
    )

    assert attrs["stride"] == "[1, 1]"


def test_build_node_attrs_uses_schema_name_for_generic_input():
    """Use the schema-derived argument name for a generic ValueNode."""
    node = OperatorNode(
        name="aten_add_tensor_",
        op="aten.add.Tensor",
        inputs=[
            ValueNode(
                name="value",
                output_shapes=[[1]],
                dtype="torch.float32",
            ),
        ],
        named_args=["other"],
    )

    attrs = key_value_attrs(build_node_attrs(node))

    assert attrs["other"] == "shape=[1], dtype=torch.float32"


def test_build_node_attrs_omits_output_shapes_when_absent():
    node = OperatorNode(
        name="custom_op",
        op="custom.op",
        output_shapes=None,
    )

    attrs = key_value_attrs(build_node_attrs(node))

    assert "Output Shapes" not in attrs


def test_build_node_attrs_omits_input_without_displayable_metadata():
    node = OperatorNode(
        name="custom_op_",
        op="custom.op",
        inputs=[ValueNode(name="unknown_value_")],
        named_args=["value"],
    )

    attrs = key_value_attrs(build_node_attrs(node))

    assert "value" not in attrs


def test_build_input_metadata_includes_shapes_constants_and_debug_handles():
    producer = OperatorNode(
        name="aten_relu_default_",
        op="aten.relu.default",
        output_shapes=[[2, 3]],
        metadata={"debug_handle": 11},
    )
    tensor_input = ValueNode(
        name="weight_",
        output_shapes=[[3, 4]],
        dtype="torch.float32",
        metadata={"debug_handle": 12},
    )
    const_input = ValueNode(name="dims_", val=[1, 0])

    metadata_items = build_input_metadata(
        [producer, tensor_input, const_input],
        named_args=["self", "weight"],
    )

    assert len(metadata_items) == 3
    assert key_value_attrs(metadata_items[0].attrs) == {
        "Source Node": "aten_relu_default_",
        "Arg Name": "self",
        "Shape": "[2, 3]",
        "Debug Handle": "11",
    }
    assert key_value_attrs(metadata_items[1].attrs) == {
        "Source Node": "weight_",
        "Arg Name": "weight",
        "Shape": "[3, 4]",
        "DType": "torch.float32",
        "Debug Handle": "12",
    }
    assert key_value_attrs(metadata_items[2].attrs) == {
        "Source Node": "dims_",
        "Const Value": "[1, 0]",
    }


def test_build_output_metadata_omits_optional_attrs_when_absent():
    node = OperatorNode(
        name="aten_relu_default_",
        op="aten.relu.default",
        output_shapes=[[4, 5]],
    )

    metadata_items = build_output_metadata(node)

    assert len(metadata_items) == 1
    assert metadata_items[0].id == "0"
    assert key_value_attrs(metadata_items[0].attrs) == {
        "Shape": "[4, 5]",
    }


def test_build_output_metadata_includes_dtype_and_debug_handle_when_present():
    node = ValueNode(
        name="x_",
        output_shapes=[[1, 3, 32, 32]],
        dtype="torch.float32",
        metadata={"debug_handle": 7},
    )

    metadata_items = build_output_metadata(node, id_prefix="x_")

    assert len(metadata_items) == 1
    assert metadata_items[0].id == "x_"
    assert key_value_attrs(metadata_items[0].attrs) == {
        "Shape": "[1, 3, 32, 32]",
        "DType": "torch.float32",
        "Debug Handle": "7",
    }


def test_build_output_metadata_omits_items_without_shapes():
    node = OperatorNode(
        name="custom_op",
        op="custom.op",
        output_shapes=None,
    )

    assert build_output_metadata(node) == []
