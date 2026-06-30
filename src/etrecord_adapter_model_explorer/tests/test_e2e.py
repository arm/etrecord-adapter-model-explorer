# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from threading import Lock
from time import sleep
from types import SimpleNamespace

import pytest
from executorch.devtools.debug_format.base_schema import (
    OperatorGraph,
    OperatorNode,
    ValueNode,
)

from etrecord_adapter_model_explorer.builder import builder as builder_module
from etrecord_adapter_model_explorer.builder.builder import (
    BuildContext,
    ETRecordGraphBuilder,
)
from etrecord_adapter_model_explorer.builder.delegate import DelegateGrouping

TEST_FIXTURES_ROOT = Path(__file__).resolve().parent / "fixtures"
FIXTURES_ROOT = TEST_FIXTURES_ROOT / "simple_convnet"
MULTI_INPUT_FIXTURES_ROOT = TEST_FIXTURES_ROOT / "multi_input"


def _node_by_id(graph_dict: dict, node_id: str) -> dict:
    for node in graph_dict["nodes"]:
        if node["id"] == node_id:
            return node
    raise AssertionError(f"Node {node_id!r} was not found in graph output.")


def _attrs_dict(node_dict: dict) -> dict[str, str]:
    return {attr["key"]: attr["value"] for attr in node_dict["attrs"]}


def _metadata_attrs(
    node_dict: dict, metadata_key: str, item_id: str
) -> dict[str, str]:
    for item in node_dict[metadata_key]:
        if item["id"] == item_id:
            return {attr["key"]: attr["value"] for attr in item["attrs"]}
    raise AssertionError(
        f"Metadata item {item_id!r} was not found in {metadata_key!r}."
    )


def _empty_build_context() -> BuildContext:
    return BuildContext(
        delegate_namespace_by_handle={},
        delegate_group_attributes={},
        parameter_state_dict={},
        placeholder_to_state_name={},
        named_arg_values_by_node={},
    )


def test_etrecord_builds_graph():
    builder = ETRecordGraphBuilder(
        FIXTURES_ROOT / "input.etrecord", settings={}
    )
    graph = asdict(builder.build()[0].graphs[0])

    assert graph["id"] == "forward"
    assert [node["id"] for node in graph["nodes"]] == [
        "GraphInputs",
        "aten_convolution_default_",
        "aten_relu_default_",
        "aten_convolution_default_1_",
        "aten_relu_default_1_",
        "aten_mean_dim_",
        "aten_view_copy_default_",
        "aten_permute_copy_default_",
        "aten_addmm_default_",
        "GraphOutputs",
    ]

    conv1 = _node_by_id(graph, "aten_convolution_default_")
    conv1_attrs = _attrs_dict(conv1)

    assert conv1["label"] == "Conv2d"
    assert conv1["namespace"] == "conv1"
    assert conv1["style"] is None
    assert conv1_attrs["debug_handle"] == "1"
    assert conv1_attrs["Op Type"] == "aten.convolution.default"
    assert conv1_attrs["weight"].startswith("shape=[8, 3, 3, 3]")
    assert conv1_attrs["bias"].startswith("shape=[8]")
    assert conv1_attrs["stride"] == "[1, 1]"
    assert conv1_attrs["padding"] == "[1, 1]"
    assert "nn_module_stack" not in conv1_attrs
    assert "stack_trace" not in conv1_attrs
    assert "Named Args" not in conv1_attrs
    assert "Runtime Avg (ms)" not in conv1_attrs
    assert conv1["incomingEdges"] == [
        {
            "sourceNodeId": "GraphInputs",
            "sourceNodeOutputId": "x_",
            "targetNodeInputId": "0",
        }
    ]

    graph_inputs = _node_by_id(graph, "GraphInputs")
    assert graph_inputs["outputsMetadata"] == [
        {
            "id": "x_",
            "attrs": [
                {"key": "Shape", "value": "[1, 3, 32, 32]"},
                {"key": "DType", "value": "[torch.float32]"},
            ],
        }
    ]

    conv1_weight = _metadata_attrs(conv1, "inputsMetadata", "1")
    conv1_stride = _metadata_attrs(conv1, "inputsMetadata", "3")
    conv1_output = _metadata_attrs(conv1, "outputsMetadata", "0")
    assert conv1_weight == {
        "Source Node": "p_conv1_weight_",
        "Arg Name": "weight",
        "Shape": "[8, 3, 3, 3]",
        "DType": "[torch.float32]",
    }
    assert conv1_stride == {
        "Source Node": "stride_0",
        "Arg Name": "stride",
        "Const Value": "[1, 1]",
    }
    assert conv1_output == {
        "Shape": "[1, 8, 32, 32]",
        "Debug Handle": "1",
    }

    mean = _node_by_id(graph, "aten_mean_dim_")
    mean_attrs = _attrs_dict(mean)
    assert mean["label"] == "AdaptiveAvgPool2d"
    assert mean["namespace"] == "pool"
    assert mean_attrs["keepdim"] == "True"
    assert mean_attrs["dtype"] == "None"
    assert _metadata_attrs(mean, "inputsMetadata", "1") == {
        "Source Node": "dim_12",
        "Arg Name": "dim",
        "Const Value": "[-1, -2]",
    }

    outputs = _node_by_id(graph, "GraphOutputs")
    assert outputs["inputsMetadata"] == [
        {
            "id": "0",
            "attrs": [
                {"key": "Source Node", "value": "aten_addmm_default_"},
                {"key": "Shape", "value": "[1, 10]"},
                {"key": "Debug Handle", "value": "7"},
            ],
        }
    ]

    assert outputs["incomingEdges"] == [
        {
            "sourceNodeId": "aten_addmm_default_",
            "sourceNodeOutputId": "0",
            "targetNodeInputId": "0",
        }
    ]


def test_etrecord_copy_builds_without_external_files(tmp_path):
    etrecord_path = tmp_path / "simple_convnet.etrecord"
    etrecord_path.write_bytes((FIXTURES_ROOT / "input.etrecord").read_bytes())

    builder = ETRecordGraphBuilder(etrecord_path, settings={})
    graph = asdict(builder.build()[0].graphs[0])

    conv1 = _node_by_id(graph, "aten_convolution_default_")
    conv1_attrs = _attrs_dict(conv1)

    assert "Runtime Avg (ms)" not in conv1_attrs
    assert conv1["style"] is None


def test_concurrent_etrecord_loads_are_serialized(monkeypatch):
    active_calls = 0
    max_active_calls = 0
    active_calls_lock = Lock()
    edge_graph = OperatorGraph(
        graph_name="base",
        elements=[
            OperatorGraph(graph_name="inputs", elements=[]),
            OperatorGraph(graph_name="forward", elements=[]),
            OperatorGraph(graph_name="outputs", elements=[]),
        ],
    )
    edge_dialect_program = SimpleNamespace(
        graph=SimpleNamespace(nodes=[]),
        graph_signature=SimpleNamespace(
            inputs_to_parameters={},
            inputs_to_buffers={},
            inputs_to_lifted_tensor_constants={},
        ),
        state_dict={},
    )
    etrecord = SimpleNamespace(edge_dialect_program=edge_dialect_program)

    def parse_etrecord(path: str):
        del path
        nonlocal active_calls, max_active_calls
        with active_calls_lock:
            active_calls += 1
            max_active_calls = max(max_active_calls, active_calls)
        sleep(0.05)
        with active_calls_lock:
            active_calls -= 1
        return etrecord

    monkeypatch.setattr(builder_module, "parse_etrecord", parse_etrecord)
    monkeypatch.setattr(
        builder_module,
        "gen_graphs_from_etrecord",
        lambda _: {builder_module.EDGE_DIALECT_GRAPH_KEY: edge_graph},
    )
    monkeypatch.setattr(
        builder_module.ETRecordDelegateBuilder,
        "build",
        lambda _: DelegateGrouping(
            namespace_by_handle={},
            group_attributes={},
        ),
    )

    builders = [
        ETRecordGraphBuilder(FIXTURES_ROOT / "input.etrecord", settings={})
        for _ in range(3)
    ]
    with ThreadPoolExecutor(max_workers=3) as executor:
        contexts = list(
            executor.map(lambda builder: builder._load_context(), builders)
        )

    assert len(contexts) == 3
    assert max_active_calls == 1


def test_etrecord_connections_preserve_original_input_indices():
    builder = ETRecordGraphBuilder(
        FIXTURES_ROOT / "input.etrecord", settings={}
    )
    graph = asdict(builder.build()[0].graphs[0])

    addmm = _node_by_id(graph, "aten_addmm_default_")
    permute = _node_by_id(graph, "aten_permute_copy_default_")

    assert permute["incomingEdges"] == []
    assert _metadata_attrs(permute, "inputsMetadata", "0") == {
        "Source Node": "p_fc_weight_",
        "Arg Name": "self",
        "Shape": "[10, 16]",
        "DType": "[torch.float32]",
    }
    assert _metadata_attrs(permute, "inputsMetadata", "1") == {
        "Source Node": "dims_14",
        "Arg Name": "dims",
        "Const Value": "[1, 0]",
    }

    assert addmm["incomingEdges"] == [
        {
            "sourceNodeId": "aten_view_copy_default_",
            "sourceNodeOutputId": "0",
            "targetNodeInputId": "1",
        },
        {
            "sourceNodeId": "aten_permute_copy_default_",
            "sourceNodeOutputId": "0",
            "targetNodeInputId": "2",
        },
    ]
    assert _metadata_attrs(addmm, "inputsMetadata", "0") == {
        "Source Node": "p_fc_bias_",
        "Arg Name": "self",
        "Shape": "[10]",
        "DType": "[torch.float32]",
    }
    assert _metadata_attrs(addmm, "inputsMetadata", "1") == {
        "Source Node": "aten_view_copy_default_",
        "Arg Name": "mat1",
        "Shape": "[1, 16]",
        "Debug Handle": "6",
    }
    assert _metadata_attrs(addmm, "inputsMetadata", "2") == {
        "Source Node": "aten_permute_copy_default_",
        "Arg Name": "mat2",
        "Shape": "[16, 10]",
        "Debug Handle": "7",
    }


def test_multi_input_etrecord_builds_expected_branching_graph():
    builder = ETRecordGraphBuilder(
        MULTI_INPUT_FIXTURES_ROOT / "input.etrecord", settings={}
    )
    graph = asdict(builder.build()[0].graphs[0])

    assert [node["id"] for node in graph["nodes"]] == [
        "GraphInputs",
        "aten_convolution_default_",
        "aten_relu_default_",
        "aten_mean_dim_",
        "aten_add_tensor_",
        "aten_sigmoid_default_",
        "GraphOutputs",
    ]

    graph_inputs = _node_by_id(graph, "GraphInputs")
    assert graph_inputs["outputsMetadata"] == [
        {
            "id": "image_",
            "attrs": [
                {"key": "Shape", "value": "[2, 3, 16, 16]"},
                {"key": "DType", "value": "[torch.float32]"},
            ],
        },
        {
            "id": "weight_",
            "attrs": [
                {"key": "Shape", "value": "[4, 3, 3, 3]"},
                {"key": "DType", "value": "[torch.float32]"},
            ],
        },
    ]

    conv = _node_by_id(graph, "aten_convolution_default_")
    conv_attrs = _attrs_dict(conv)
    assert conv["namespace"] == ""
    assert conv["label"] == "aten::convolution"
    assert conv_attrs["Op Type"] == "aten.convolution.default"
    assert conv["incomingEdges"] == [
        {
            "sourceNodeId": "GraphInputs",
            "sourceNodeOutputId": "image_",
            "targetNodeInputId": "0",
        },
        {
            "sourceNodeId": "GraphInputs",
            "sourceNodeOutputId": "weight_",
            "targetNodeInputId": "1",
        },
    ]
    assert _metadata_attrs(conv, "inputsMetadata", "0") == {
        "Source Node": "image_",
        "Arg Name": "input",
        "Shape": "[2, 3, 16, 16]",
        "DType": "[torch.float32]",
    }
    assert _metadata_attrs(conv, "inputsMetadata", "1") == {
        "Source Node": "weight_",
        "Arg Name": "weight",
        "Shape": "[4, 3, 3, 3]",
        "DType": "[torch.float32]",
    }

    add = _node_by_id(graph, "aten_add_tensor_")
    add_attrs = _attrs_dict(add)
    assert add_attrs["Op Type"] == "aten.add.Tensor"
    assert add_attrs["alpha"] == "1"
    assert add["incomingEdges"] == [
        {
            "sourceNodeId": "aten_relu_default_",
            "sourceNodeOutputId": "0",
            "targetNodeInputId": "0",
        },
        {
            "sourceNodeId": "aten_mean_dim_",
            "sourceNodeOutputId": "0",
            "targetNodeInputId": "1",
        },
    ]
    assert _metadata_attrs(add, "inputsMetadata", "0") == {
        "Source Node": "aten_relu_default_",
        "Arg Name": "self",
        "Shape": "[2, 4, 16, 16]",
        "Debug Handle": "2",
    }
    assert _metadata_attrs(add, "inputsMetadata", "1") == {
        "Source Node": "aten_mean_dim_",
        "Arg Name": "other",
        "Shape": "[2, 4, 1, 1]",
        "Debug Handle": "3",
    }

    sigmoid = _node_by_id(graph, "aten_sigmoid_default_")
    sigmoid_attrs = _attrs_dict(sigmoid)
    assert sigmoid_attrs["Op Type"] == "aten.sigmoid.default"
    assert sigmoid["incomingEdges"] == [
        {
            "sourceNodeId": "aten_add_tensor_",
            "sourceNodeOutputId": "0",
            "targetNodeInputId": "0",
        }
    ]

    pooled = _node_by_id(graph, "aten_mean_dim_")
    pooled_attrs = _attrs_dict(pooled)
    assert pooled["label"] == "AdaptiveAvgPool2d"
    assert pooled["namespace"] == "pool"
    assert pooled_attrs["Op Type"] == "aten.mean.dim"
    assert _metadata_attrs(pooled, "inputsMetadata", "1") == {
        "Source Node": "dim_6",
        "Arg Name": "dim",
        "Const Value": "[-1, -2]",
    }

    outputs = _node_by_id(graph, "GraphOutputs")
    assert outputs["incomingEdges"] == [
        {
            "sourceNodeId": "aten_sigmoid_default_",
            "sourceNodeOutputId": "0",
            "targetNodeInputId": "0",
        }
    ]
    assert outputs["inputsMetadata"] == [
        {
            "id": "0",
            "attrs": [
                {"key": "Source Node", "value": "aten_sigmoid_default_"},
                {"key": "Shape", "value": "[2, 4, 16, 16]"},
                {"key": "Debug Handle", "value": "5"},
            ],
        }
    ]


def test_graph_outputs_support_multiple_operator_outputs_and_passthrough_values():
    input_value = ValueNode(
        name="x_",
        output_shapes=[[1, 4]],
        dtype="torch.float32",
        metadata={"debug_handle": 10},
    )
    split_op = OperatorNode(
        name="aten_split_copy_tensor_",
        op="aten.split_copy.Tensor",
        inputs=[input_value],
        output_shapes=[[1, 1], [1, 3]],
        metadata={"debug_handle": 20},
    )
    edge_graph = OperatorGraph(
        graph_name="base",
        elements=[
            OperatorGraph(graph_name="inputs", elements=[input_value]),
            OperatorGraph(graph_name="forward", elements=[split_op]),
            OperatorGraph(
                graph_name="outputs",
                elements=[
                    ValueNode(
                        name="first_output_",
                        inputs=[split_op],
                        output_shapes=[[1, 1]],
                    ),
                    ValueNode(
                        name="second_output_",
                        inputs=[split_op],
                        output_shapes=[[1, 3]],
                    ),
                    ValueNode(
                        name="passthrough_output_",
                        inputs=[input_value],
                        output_shapes=[[1, 4]],
                    ),
                ],
            ),
        ],
    )

    graph = asdict(
        ETRecordGraphBuilder._build_model_explorer_graph(
            Path("synthetic.etrecord"),
            edge_graph,
            _empty_build_context(),
        )
    )
    split = _node_by_id(graph, "aten_split_copy_tensor_")
    outputs = _node_by_id(graph, "GraphOutputs")

    assert [item["id"] for item in split["outputsMetadata"]] == [
        "0:0",
        "0:1",
    ]
    assert outputs["incomingEdges"] == [
        {
            "sourceNodeId": "aten_split_copy_tensor_",
            "sourceNodeOutputId": "0:0",
            "targetNodeInputId": "0",
        },
        {
            "sourceNodeId": "aten_split_copy_tensor_",
            "sourceNodeOutputId": "0:1",
            "targetNodeInputId": "1",
        },
        {
            "sourceNodeId": "GraphInputs",
            "sourceNodeOutputId": "x_",
            "targetNodeInputId": "2",
        },
    ]
    assert outputs["inputsMetadata"] == [
        {
            "id": "0",
            "attrs": [
                {"key": "Source Node", "value": "aten_split_copy_tensor_"},
                {"key": "Shape", "value": "[1, 1]"},
                {"key": "Debug Handle", "value": "20"},
            ],
        },
        {
            "id": "1",
            "attrs": [
                {"key": "Source Node", "value": "aten_split_copy_tensor_"},
                {"key": "Shape", "value": "[1, 3]"},
                {"key": "Debug Handle", "value": "20"},
            ],
        },
        {
            "id": "2",
            "attrs": [
                {"key": "Source Node", "value": "x_"},
                {"key": "Shape", "value": "[1, 4]"},
                {"key": "DType", "value": "torch.float32"},
                {"key": "Debug Handle", "value": "10"},
            ],
        },
    ]


def test_etdp_input_is_rejected(tmp_path):
    invalid_path = tmp_path / "invalid.etdp"
    invalid_path.write_bytes(b"not an etrecord")
    with pytest.raises(ValueError, match=r"Expected \.etrecord"):
        ETRecordGraphBuilder(invalid_path, settings={})
