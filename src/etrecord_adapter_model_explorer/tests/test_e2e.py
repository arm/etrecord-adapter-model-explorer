# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

import unittest
import warnings
from collections import Counter
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
    ETRecordGraphBuilder,
)
from etrecord_adapter_model_explorer.builder.delegate import DelegateGrouping
from etrecord_adapter_model_explorer.tests.helpers import empty_build_context

_ASSERT = unittest.TestCase()

TEST_FIXTURES_ROOT = Path(__file__).resolve().parent / "fixtures"
FIXTURES_ROOT = TEST_FIXTURES_ROOT / "simple_convnet"
MULTI_INPUT_FIXTURES_ROOT = TEST_FIXTURES_ROOT / "multi_input"
CORE_ML_FIXTURES_ROOT = TEST_FIXTURES_ROOT / "mv2_core_ml"
EDGE_CASE_FIXTURES_ROOT = TEST_FIXTURES_ROOT / "edge_case_tensors"


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


def test_etrecord_builds_graph():
    builder = ETRecordGraphBuilder(
        FIXTURES_ROOT / "input.etrecord", settings={}
    )
    graph = asdict(builder.build()[0].graphs[0])

    _ASSERT.assertEqual(graph["id"], "forward")
    _ASSERT.assertEqual(
        [node["id"] for node in graph["nodes"]],
        [
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
        ],
    )

    conv1 = _node_by_id(graph, "aten_convolution_default_")
    conv1_attrs = _attrs_dict(conv1)

    _ASSERT.assertEqual(conv1["label"], "Conv2d")
    _ASSERT.assertEqual(conv1["namespace"], "conv1")
    _ASSERT.assertIsNone(conv1["style"])
    _ASSERT.assertEqual(conv1_attrs["debug_handle"], "1")
    _ASSERT.assertEqual(conv1_attrs["Op Type"], "aten.convolution.default")
    _ASSERT.assertTrue(conv1_attrs["weight"].startswith("shape=[8, 3, 3, 3]"))
    _ASSERT.assertTrue(conv1_attrs["bias"].startswith("shape=[8]"))
    _ASSERT.assertEqual(conv1_attrs["stride"], "[1, 1]")
    _ASSERT.assertEqual(conv1_attrs["padding"], "[1, 1]")
    _ASSERT.assertNotIn("nn_module_stack", conv1_attrs)
    _ASSERT.assertNotIn("stack_trace", conv1_attrs)
    _ASSERT.assertNotIn("Named Args", conv1_attrs)
    _ASSERT.assertNotIn("Runtime Avg (ms)", conv1_attrs)
    _ASSERT.assertEqual(
        conv1["incomingEdges"],
        [
            {
                "sourceNodeId": "GraphInputs",
                "sourceNodeOutputId": "x_",
                "targetNodeInputId": "0",
            }
        ],
    )

    graph_inputs = _node_by_id(graph, "GraphInputs")
    _ASSERT.assertEqual(
        graph_inputs["outputsMetadata"],
        [
            {
                "id": "x_",
                "attrs": [
                    {"key": "Shape", "value": "[1, 3, 32, 32]"},
                    {"key": "DType", "value": "[torch.float32]"},
                ],
            }
        ],
    )

    conv1_weight = _metadata_attrs(conv1, "inputsMetadata", "1")
    conv1_stride = _metadata_attrs(conv1, "inputsMetadata", "3")
    conv1_output = _metadata_attrs(conv1, "outputsMetadata", "0")
    _ASSERT.assertEqual(
        conv1_weight,
        {
            "Source Node": "p_conv1_weight_",
            "Arg Name": "weight",
            "Shape": "[8, 3, 3, 3]",
            "DType": "[torch.float32]",
        },
    )
    _ASSERT.assertEqual(
        conv1_stride,
        {
            "Source Node": "stride_0",
            "Arg Name": "stride",
            "Const Value": "[1, 1]",
        },
    )
    _ASSERT.assertEqual(
        conv1_output, {"Shape": "[1, 8, 32, 32]", "Debug Handle": "1"}
    )

    mean = _node_by_id(graph, "aten_mean_dim_")
    mean_attrs = _attrs_dict(mean)
    _ASSERT.assertEqual(mean["label"], "AdaptiveAvgPool2d")
    _ASSERT.assertEqual(mean["namespace"], "pool")
    _ASSERT.assertEqual(mean_attrs["keepdim"], "True")
    _ASSERT.assertEqual(mean_attrs["dtype"], "None")
    _ASSERT.assertEqual(
        _metadata_attrs(mean, "inputsMetadata", "1"),
        {
            "Source Node": "dim_12",
            "Arg Name": "dim",
            "Const Value": "[-1, -2]",
        },
    )

    outputs = _node_by_id(graph, "GraphOutputs")
    _ASSERT.assertEqual(
        outputs["inputsMetadata"],
        [
            {
                "id": "0",
                "attrs": [
                    {"key": "Source Node", "value": "aten_addmm_default_"},
                    {"key": "Shape", "value": "[1, 10]"},
                    {"key": "Debug Handle", "value": "7"},
                ],
            }
        ],
    )

    _ASSERT.assertEqual(
        outputs["incomingEdges"],
        [
            {
                "sourceNodeId": "aten_addmm_default_",
                "sourceNodeOutputId": "0",
                "targetNodeInputId": "0",
            }
        ],
    )


def test_etrecord_copy_builds_without_external_files(tmp_path):
    etrecord_path = tmp_path / "simple_convnet.etrecord"
    etrecord_path.write_bytes((FIXTURES_ROOT / "input.etrecord").read_bytes())

    builder = ETRecordGraphBuilder(etrecord_path, settings={})
    graph = asdict(builder.build()[0].graphs[0])

    conv1 = _node_by_id(graph, "aten_convolution_default_")
    conv1_attrs = _attrs_dict(conv1)

    _ASSERT.assertNotIn("Runtime Avg (ms)", conv1_attrs)
    _ASSERT.assertIsNone(conv1["style"])


def test_bin_etrecord_copy_builds_without_external_files(tmp_path):
    """An ETRecord using the .bin filename convention is accepted."""
    etrecord_path = tmp_path / "simple_convnet.bin"
    etrecord_path.write_bytes((FIXTURES_ROOT / "input.etrecord").read_bytes())

    graph = asdict(
        ETRecordGraphBuilder(etrecord_path, settings={}).build()[0].graphs[0]
    )

    _ASSERT.assertEqual(graph["id"], "forward")


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
            executor.map(
                lambda builder: builder._load_graph_contexts(), builders
            )
        )

    _ASSERT.assertEqual(len(contexts), 3)
    _ASSERT.assertEqual(max_active_calls, 1)


def test_etrecord_builds_all_generated_graphs(monkeypatch, tmp_path):
    etrecord_path = tmp_path / "multi_graph.etrecord"
    etrecord_path.write_bytes(b"synthetic etrecord")
    exported_program = _empty_exported_program()
    etrecord = SimpleNamespace(
        edge_dialect_program={
            "forward": exported_program,
            "encode": exported_program,
        },
        graph_map={},
    )
    operator_graphs = {
        builder_module.EDGE_DIALECT_GRAPH_KEY: _synthetic_edge_graph(
            "aten_relu_default_"
        ),
        "edge_dialect_exported_program/encode": _synthetic_edge_graph(
            "aten_sigmoid_default_"
        ),
    }

    monkeypatch.setattr(builder_module, "parse_etrecord", lambda _: etrecord)
    monkeypatch.setattr(
        builder_module.ETRecordGraphBuilder,
        "_gen_operator_graphs",
        staticmethod(lambda _: operator_graphs),
    )
    monkeypatch.setattr(
        builder_module.ETRecordDelegateBuilder,
        "build",
        lambda _: DelegateGrouping(
            namespace_by_handle={},
            group_attributes={},
        ),
    )

    collection = asdict(
        ETRecordGraphBuilder(etrecord_path, settings={}).build()[0]
    )

    _ASSERT.assertEqual(collection["label"], "multi_graph.etrecord")
    _ASSERT.assertEqual(
        [graph["id"] for graph in collection["graphs"]], ["forward", "encode"]
    )
    _ASSERT.assertEqual(
        [
            _node_by_id(graph, graph_node_id)["id"]
            for graph, graph_node_id in zip(
                collection["graphs"],
                ["aten_relu_default_", "aten_sigmoid_default_"],
                strict=True,
            )
        ],
        ["aten_relu_default_", "aten_sigmoid_default_"],
    )


def _empty_exported_program() -> SimpleNamespace:
    return SimpleNamespace(
        graph=SimpleNamespace(nodes=[]),
        graph_signature=SimpleNamespace(
            inputs_to_parameters={},
            inputs_to_buffers={},
            inputs_to_lifted_tensor_constants={},
        ),
        state_dict={},
    )


def _synthetic_edge_graph(operator_name: str) -> OperatorGraph:
    input_value = ValueNode(
        name=f"{operator_name}_input",
        output_shapes=[[1]],
        dtype="torch.float32",
    )
    operator = OperatorNode(
        name=operator_name,
        op=operator_name.removesuffix("_").replace("_", "."),
        inputs=[input_value],
        output_shapes=[[1]],
        metadata={"debug_handle": 1},
    )
    output_value = ValueNode(
        name=f"{operator_name}_output",
        inputs=[operator],
        output_shapes=[[1]],
    )
    return OperatorGraph(
        graph_name="base",
        elements=[
            OperatorGraph(graph_name="inputs", elements=[input_value]),
            OperatorGraph(graph_name="forward", elements=[operator]),
            OperatorGraph(graph_name="outputs", elements=[output_value]),
        ],
    )


def test_etrecord_connections_preserve_original_input_indices():
    builder = ETRecordGraphBuilder(
        FIXTURES_ROOT / "input.etrecord", settings={}
    )
    graph = asdict(builder.build()[0].graphs[0])

    addmm = _node_by_id(graph, "aten_addmm_default_")
    permute = _node_by_id(graph, "aten_permute_copy_default_")

    _ASSERT.assertEqual(permute["incomingEdges"], [])
    _ASSERT.assertEqual(
        _metadata_attrs(permute, "inputsMetadata", "0"),
        {
            "Source Node": "p_fc_weight_",
            "Arg Name": "self",
            "Shape": "[10, 16]",
            "DType": "[torch.float32]",
        },
    )
    _ASSERT.assertEqual(
        _metadata_attrs(permute, "inputsMetadata", "1"),
        {
            "Source Node": "dims_14",
            "Arg Name": "dims",
            "Const Value": "[1, 0]",
        },
    )

    _ASSERT.assertEqual(
        addmm["incomingEdges"],
        [
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
        ],
    )
    _ASSERT.assertEqual(
        _metadata_attrs(addmm, "inputsMetadata", "0"),
        {
            "Source Node": "p_fc_bias_",
            "Arg Name": "self",
            "Shape": "[10]",
            "DType": "[torch.float32]",
        },
    )
    _ASSERT.assertEqual(
        _metadata_attrs(addmm, "inputsMetadata", "1"),
        {
            "Source Node": "aten_view_copy_default_",
            "Arg Name": "mat1",
            "Shape": "[1, 16]",
            "Debug Handle": "6",
        },
    )
    _ASSERT.assertEqual(
        _metadata_attrs(addmm, "inputsMetadata", "2"),
        {
            "Source Node": "aten_permute_copy_default_",
            "Arg Name": "mat2",
            "Shape": "[16, 10]",
            "Debug Handle": "7",
        },
    )


def test_multi_input_etrecord_builds_expected_branching_graph():
    builder = ETRecordGraphBuilder(
        MULTI_INPUT_FIXTURES_ROOT / "input.etrecord", settings={}
    )
    graph = asdict(builder.build()[0].graphs[0])

    _ASSERT.assertEqual(
        [node["id"] for node in graph["nodes"]],
        [
            "GraphInputs",
            "aten_convolution_default_",
            "aten_relu_default_",
            "aten_mean_dim_",
            "aten_add_tensor_",
            "aten_sigmoid_default_",
            "GraphOutputs",
        ],
    )

    graph_inputs = _node_by_id(graph, "GraphInputs")
    _ASSERT.assertEqual(
        graph_inputs["outputsMetadata"],
        [
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
        ],
    )

    conv = _node_by_id(graph, "aten_convolution_default_")
    conv_attrs = _attrs_dict(conv)
    _ASSERT.assertEqual(conv["namespace"], "")
    _ASSERT.assertEqual(conv["label"], "aten::convolution")
    _ASSERT.assertEqual(conv_attrs["Op Type"], "aten.convolution.default")
    _ASSERT.assertEqual(
        conv["incomingEdges"],
        [
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
        ],
    )
    _ASSERT.assertEqual(
        _metadata_attrs(conv, "inputsMetadata", "0"),
        {
            "Source Node": "image_",
            "Arg Name": "input",
            "Shape": "[2, 3, 16, 16]",
            "DType": "[torch.float32]",
        },
    )
    _ASSERT.assertEqual(
        _metadata_attrs(conv, "inputsMetadata", "1"),
        {
            "Source Node": "weight_",
            "Arg Name": "weight",
            "Shape": "[4, 3, 3, 3]",
            "DType": "[torch.float32]",
        },
    )

    add = _node_by_id(graph, "aten_add_tensor_")
    add_attrs = _attrs_dict(add)
    _ASSERT.assertEqual(add_attrs["Op Type"], "aten.add.Tensor")
    _ASSERT.assertEqual(add_attrs["alpha"], "1")
    _ASSERT.assertEqual(
        add["incomingEdges"],
        [
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
        ],
    )
    _ASSERT.assertEqual(
        _metadata_attrs(add, "inputsMetadata", "0"),
        {
            "Source Node": "aten_relu_default_",
            "Arg Name": "self",
            "Shape": "[2, 4, 16, 16]",
            "Debug Handle": "2",
        },
    )
    _ASSERT.assertEqual(
        _metadata_attrs(add, "inputsMetadata", "1"),
        {
            "Source Node": "aten_mean_dim_",
            "Arg Name": "other",
            "Shape": "[2, 4, 1, 1]",
            "Debug Handle": "3",
        },
    )

    sigmoid = _node_by_id(graph, "aten_sigmoid_default_")
    sigmoid_attrs = _attrs_dict(sigmoid)
    _ASSERT.assertEqual(sigmoid_attrs["Op Type"], "aten.sigmoid.default")
    _ASSERT.assertEqual(
        sigmoid["incomingEdges"],
        [
            {
                "sourceNodeId": "aten_add_tensor_",
                "sourceNodeOutputId": "0",
                "targetNodeInputId": "0",
            }
        ],
    )

    pooled = _node_by_id(graph, "aten_mean_dim_")
    pooled_attrs = _attrs_dict(pooled)
    _ASSERT.assertEqual(pooled["label"], "AdaptiveAvgPool2d")
    _ASSERT.assertEqual(pooled["namespace"], "pool")
    _ASSERT.assertEqual(pooled_attrs["Op Type"], "aten.mean.dim")
    _ASSERT.assertEqual(
        _metadata_attrs(pooled, "inputsMetadata", "1"),
        {"Source Node": "dim_6", "Arg Name": "dim", "Const Value": "[-1, -2]"},
    )

    outputs = _node_by_id(graph, "GraphOutputs")
    _ASSERT.assertEqual(
        outputs["incomingEdges"],
        [
            {
                "sourceNodeId": "aten_sigmoid_default_",
                "sourceNodeOutputId": "0",
                "targetNodeInputId": "0",
            }
        ],
    )
    _ASSERT.assertEqual(
        outputs["inputsMetadata"],
        [
            {
                "id": "0",
                "attrs": [
                    {"key": "Source Node", "value": "aten_sigmoid_default_"},
                    {"key": "Shape", "value": "[2, 4, 16, 16]"},
                    {"key": "Debug Handle", "value": "5"},
                ],
            }
        ],
    )


@pytest.fixture(scope="module")
def edge_case_tensor_graph():
    builder = ETRecordGraphBuilder(
        EDGE_CASE_FIXTURES_ROOT / "edge_case_tensors.etrecord", settings={}
    )
    return asdict(builder.build()[0].graphs[0])


@pytest.mark.parametrize(
    ("node_id", "attribute_name", "expected_value"),
    [
        (
            "dim_order_ops__clone_dim_order_default_",
            "empty_tensor",
            "shape=[0, 3], dtype=torch.float32, "
            "values=tensor([], size=(0, 3))",
        ),
        (
            "dim_order_ops__clone_dim_order_default_1_",
            "boolean_tensor",
            "shape=[2], dtype=torch.bool, values=[True, False]",
        ),
        (
            "dim_order_ops__clone_dim_order_default_2_",
            "complex_tensor",
            "shape=[1], dtype=torch.complex64, values=[(1+2j)]",
        ),
    ],
)
def test_edge_case_tensor_etrecord_renders_state_dict_value(
    edge_case_tensor_graph,
    node_id,
    attribute_name,
    expected_value,
):
    _ASSERT.assertEqual(
        _attrs_dict(_node_by_id(edge_case_tensor_graph, node_id))[
            attribute_name
        ],
        expected_value,
    )


def test_edge_case_tensor_etrecord_constructs_all_outputs(
    edge_case_tensor_graph,
):
    graph = edge_case_tensor_graph
    _ASSERT.assertEqual(
        {
            node["id"]
            for node in graph["nodes"]
            if node["id"].startswith("dim_order_ops__clone_dim_order_default")
        },
        {
            "dim_order_ops__clone_dim_order_default_",
            "dim_order_ops__clone_dim_order_default_1_",
            "dim_order_ops__clone_dim_order_default_2_",
        },
    )
    _ASSERT.assertEqual(
        len(_node_by_id(graph, "GraphOutputs")["incomingEdges"]), 4
    )  # x_ input passthrough, 3 clone outputs


@pytest.mark.parametrize(
    ("node_id", "expected_attrs"),
    [
        (
            "dim_order_ops__clone_dim_order_default_",
            {"Shape": "[0, 3]", "Debug Handle": "0"},
        ),
        (
            "dim_order_ops__clone_dim_order_default_1_",
            {"Shape": "[2]", "Debug Handle": "1"},
        ),
        (
            "dim_order_ops__clone_dim_order_default_2_",
            {"Shape": "[1]", "Debug Handle": "2"},
        ),
    ],
)
def test_edge_case_tensor_etrecord_preserves_operator_output_metadata(
    edge_case_tensor_graph,
    node_id,
    expected_attrs,
):
    operator = _node_by_id(edge_case_tensor_graph, node_id)
    _ASSERT.assertEqual(
        operator["outputsMetadata"],
        [
            {
                "id": "0",
                "attrs": [
                    {"key": key, "value": value}
                    for key, value in expected_attrs.items()
                ],
            }
        ],
    )


def test_core_ml_etrecord_builds_multi_output_delegate_graph():
    # Core ML MobileNetV2 emits expected parser warnings for unsupported
    # hardtanh (min_val/max_val) and mean (keepdim) kwargs
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message=(
                r"Unsupported kwarg encountered: "
                r"aten_(?:hardtanh_default(?:_\d+)?|mean_dim), .*"
            ),
        )
        builder = ETRecordGraphBuilder(
            CORE_ML_FIXTURES_ROOT / "mv2_core_ml.etrecord", settings={}
        )
        graph = asdict(builder.build()[0].graphs[0])

    graph_inputs = _node_by_id(graph, "GraphInputs")
    graph_outputs = _node_by_id(graph, "GraphOutputs")
    _ASSERT.assertEqual(len(graph_inputs["outputsMetadata"]), 1)
    _ASSERT.assertEqual(len(graph_outputs["inputsMetadata"]), 1)
    _ASSERT.assertEqual(len(graph_outputs["incomingEdges"]), 1)

    batch_norm = _node_by_id(
        graph,
        "aten__native_batch_norm_legit_no_training_default_",
    )
    _ASSERT.assertEqual(
        batch_norm["namespace"], "delegates/CoreMLBackend/partition_0"
    )
    _ASSERT.assertEqual(
        [item["id"] for item in batch_norm["outputsMetadata"]],
        ["0:0", "0:1", "0:2"],
    )
    _ASSERT.assertEqual(
        [item["attrs"][0] for item in batch_norm["outputsMetadata"]],
        [
            {"key": "Shape", "value": "[1, 32, 112, 112]"},
            {"key": "Shape", "value": "[0]"},
            {"key": "Shape", "value": "[0]"},
        ],
    )

    delegate_namespace = "delegates/CoreMLBackend/partition_0"
    delegate_attrs = graph["groupNodeAttributes"][delegate_namespace]
    _ASSERT.assertEqual(delegate_attrs["Backend"], "CoreMLBackend")
    _ASSERT.assertEqual(delegate_attrs["Instruction Id"], "0")
    _ASSERT.assertEqual(delegate_attrs["Method"], "forward")
    handles = [
        int(handle) for handle in delegate_attrs["Debug Handles"].split(", ")
    ]

    # The fixture's delegate summary contains debug handles 1 through 153
    _ASSERT.assertEqual(handles, list(range(1, 154)))

    delegated_nodes = [
        node
        for node in graph["nodes"]
        if node["namespace"] == delegate_namespace
    ]

    delegated_handles = {
        int(_attrs_dict(node)["debug_handle"]) for node in delegated_nodes
    }
    _ASSERT.assertEqual(delegated_handles, set(handles))

    node_handle_counts = Counter(
        int(_attrs_dict(node)["debug_handle"]) for node in delegated_nodes
    )

    # Two graph nodes share handle 153
    expected_handle_counts = Counter(handles)
    expected_handle_counts[153] += 1
    _ASSERT.assertEqual(node_handle_counts, expected_handle_counts)

    _ASSERT.assertEqual(
        _attrs_dict(_node_by_id(graph, "aten_permute_copy_default_"))[
            "debug_handle"
        ],
        "153",
    )
    _ASSERT.assertEqual(
        _attrs_dict(_node_by_id(graph, "aten_addmm_default_"))["debug_handle"],
        "153",
    )

    _ASSERT.assertTrue(
        all(
            (
                node["namespace"] == delegate_namespace
                for node in delegated_nodes
            )
        )
    )

    _ASSERT.assertEqual(len(delegated_nodes), 154)
    _ASSERT.assertEqual(
        graph["groupNodeConfigs"],
        [{"namespaceRegex": "delegates(/.*)?", "layoutDirection": 0}],
    )


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
            empty_build_context(),
        )
    )
    split = _node_by_id(graph, "aten_split_copy_tensor_")
    outputs = _node_by_id(graph, "GraphOutputs")

    _ASSERT.assertEqual(
        [item["id"] for item in split["outputsMetadata"]], ["0:0", "0:1"]
    )
    _ASSERT.assertEqual(
        outputs["incomingEdges"],
        [
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
        ],
    )
    _ASSERT.assertEqual(
        outputs["inputsMetadata"],
        [
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
        ],
    )


def test_etrecord_file_not_found(tmp_path):
    missing_path = tmp_path / "missing.etrecord"
    with pytest.raises(FileNotFoundError, match="does not exist"):
        ETRecordGraphBuilder(source_path=missing_path, settings={})


def test_etdp_input_is_rejected(tmp_path):
    invalid_path = tmp_path / "invalid.etdp"
    invalid_path.write_bytes(b"not an etrecord")
    with pytest.raises(ValueError, match=r"Expected \.etrecord or \.bin"):
        ETRecordGraphBuilder(invalid_path, settings={})
