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


def _attrs_dict(items) -> dict[str, str]:
    return {item.key: item.value for item in items}


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

    attrs = _attrs_dict(build_node_attrs(node))

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

    attrs = _attrs_dict(
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

    attrs = _attrs_dict(
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
    assert _attrs_dict(metadata_items[0].attrs) == {
        "Source Node": "aten_relu_default_",
        "Arg Name": "self",
        "Shape": "[2, 3]",
        "Debug Handle": "11",
    }
    assert _attrs_dict(metadata_items[1].attrs) == {
        "Source Node": "weight_",
        "Arg Name": "weight",
        "Shape": "[3, 4]",
        "DType": "torch.float32",
        "Debug Handle": "12",
    }
    assert _attrs_dict(metadata_items[2].attrs) == {
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
    assert _attrs_dict(metadata_items[0].attrs) == {
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
    assert _attrs_dict(metadata_items[0].attrs) == {
        "Shape": "[1, 3, 32, 32]",
        "DType": "torch.float32",
        "Debug Handle": "7",
    }
