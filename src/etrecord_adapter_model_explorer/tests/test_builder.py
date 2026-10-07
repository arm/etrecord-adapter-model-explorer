# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, sentinel
from zipfile import ZipFile

import pytest
from executorch.devtools.debug_format.base_schema import (
    Node,
    OperatorGraph,
    OperatorNode,
    ValueNode,
)

from etrecord_adapter_model_explorer.builder import builder as builder_module
from etrecord_adapter_model_explorer.builder.builder import (
    ETRecordGraphBuilder,
)
from etrecord_adapter_model_explorer.tests.helpers import (
    empty_build_context,
    key_value_attrs,
)


def _patch_etrecord_loading(monkeypatch, etrecord, edge_graph):
    """Stub ExecuTorch deserialisation for builder unit tests."""
    monkeypatch.setattr(
        builder_module,
        "parse_etrecord",
        lambda _: etrecord,
    )
    monkeypatch.setattr(
        builder_module,
        "gen_graphs_from_etrecord",
        lambda _: {builder_module.EDGE_DIALECT_GRAPH_KEY: edge_graph},
    )


def test_graph_collections_property_delegates_to_build(tmp_path):
    source_path = tmp_path / "minimal.etrecord"
    source_path.touch()

    builder = ETRecordGraphBuilder(source_path, settings={})
    builder.build = Mock(return_value=sentinel.graph_collections)

    graph_collections = builder.graph_collections
    assert graph_collections is sentinel.graph_collections

    builder.build.assert_called_once()


def test_missing_matching_edge_program_is_rejected(tmp_path, monkeypatch):
    source_path = tmp_path / "missing_edge_program.etrecord"

    # Create a valid zip/container at the path
    with ZipFile(source_path, "w") as archive:
        archive.writestr("placeholder", b"")

    fake_etrecord = SimpleNamespace(
        edge_dialect_program=None,
        graph_map=None,
    )
    empty_edge_graph = OperatorGraph(graph_name="base", elements=[])

    _patch_etrecord_loading(monkeypatch, fake_etrecord, empty_edge_graph)

    builder = ETRecordGraphBuilder(source_path, settings={})

    with pytest.raises(
        ValueError,
        match="matching edge dialect exported program could not be found",
    ):
        builder.build()


def test_corrupt_container_is_rejected_by_etrecord_parser(tmp_path):
    source_path = tmp_path / "corrupt.etrecord"
    source_path.write_bytes(b"not a valid zip file")

    builder = ETRecordGraphBuilder(source_path, settings={})

    with pytest.raises(RuntimeError, match="Invalid etrecord file passed in"):
        builder.build()


def test_build_context_uses_dict_edge_dialect_program(
    tmp_path,
    monkeypatch,
):
    source_path = tmp_path / "minimal.etrecord"
    source_path.touch()

    empty_edge_graph = OperatorGraph(
        graph_name="base",
        elements=[],
    )

    fake_forward_program = SimpleNamespace(
        graph_signature=SimpleNamespace(
            inputs_to_parameters={"p_weight_": "weight"},
            inputs_to_buffers={},
            inputs_to_lifted_tensor_constants={},
        ),
        state_dict={"weight": "fake tensor"},
        graph=SimpleNamespace(nodes=[]),
    )

    fake_etrecord = SimpleNamespace(
        edge_dialect_program={"forward": fake_forward_program},
        graph_map=None,
    )

    _patch_etrecord_loading(monkeypatch, fake_etrecord, empty_edge_graph)
    builder = ETRecordGraphBuilder(source_path, settings={})

    loaded_etrecord, operator_graphs, graph_key = (
        builder._load_graph_contexts()
    )
    edge_graph = operator_graphs[graph_key]
    exported_program = builder._find_exported_program(
        loaded_etrecord,
        graph_key,
    )
    context = builder._build_context(
        exported_program,
        delegate_grouping=None,
    )

    assert edge_graph is empty_edge_graph
    assert context.parameter_state_dict == {"weight": "fake tensor"}
    assert context.placeholder_to_state_name == {
        "p_weight_": "weight",
    }


def test_graph_build_fails_with_missing_subgraphs():
    edge_graph = OperatorGraph(
        graph_name="base",
        elements=[
            OperatorGraph(graph_name="inputs", elements=[]),
            OperatorGraph(graph_name="forward", elements=[]),
            # No "outputs" subgraph
        ],
    )

    with pytest.raises(
        ValueError,
        match="missing one of",
    ):
        ETRecordGraphBuilder._build_model_explorer_graph(
            Path("minimal.etrecord"),
            edge_graph,
            empty_build_context(),
        )


def test_graph_build_skips_non_value_nodes():
    edge_graph = OperatorGraph(
        graph_name="base",
        elements=[
            OperatorGraph(
                graph_name="inputs",
                elements=[
                    ValueNode(name="input_", output_shapes=[[1]]),
                    ValueNode(name="weight_", output_shapes=[[2]]),
                    Node(name="not_a_value_node"),
                ],
            ),
            OperatorGraph(graph_name="forward", elements=[]),
            OperatorGraph(graph_name="outputs", elements=[]),
        ],
    )

    graph = ETRecordGraphBuilder._build_model_explorer_graph(
        Path("minimal.etrecord"),
        edge_graph,
        empty_build_context(),
    )

    graph_inputs = next(
        node for node in graph.nodes if node.id == "GraphInputs"
    )

    assert [item.id for item in graph_inputs.outputsMetadata] == [
        "input_",
        "weight_",
    ]


def test_graph_output_without_shapes_returns_no_metadata():
    edge_graph = OperatorGraph(
        graph_name="base",
        elements=[
            OperatorGraph(graph_name="inputs", elements=[]),
            OperatorGraph(graph_name="forward", elements=[]),
            OperatorGraph(
                graph_name="outputs", elements=[ValueNode(name="output_")]
            ),
        ],
    )

    graph = ETRecordGraphBuilder._build_model_explorer_graph(
        Path("minimal.etrecord"),
        edge_graph,
        empty_build_context(),
    )

    graph_outputs = next(
        node for node in graph.nodes if node.id == "GraphOutputs"
    )

    assert graph_outputs.inputsMetadata == []


def test_graph_output_with_sourceless_constant():
    edge_graph = OperatorGraph(
        graph_name="base",
        elements=[
            OperatorGraph(graph_name="inputs", elements=[]),
            OperatorGraph(graph_name="forward", elements=[]),
            OperatorGraph(
                graph_name="outputs",
                elements=[
                    ValueNode(name="output_", output_shapes=[[1]], val=1000)
                ],
            ),
        ],
    )

    graph = ETRecordGraphBuilder._build_model_explorer_graph(
        Path("minimal.etrecord"),
        edge_graph,
        empty_build_context(),
    )

    graph_outputs = next(
        node for node in graph.nodes if node.id == "GraphOutputs"
    )

    output_attrs = key_value_attrs(graph_outputs.inputsMetadata[0].attrs)

    assert output_attrs["Const Value"] == "1000"


def test_graph_output_ignores_unknown_producer():
    output_value = ValueNode(
        name="output",
        inputs=[ValueNode(name="unknown_value_")],
    )

    incoming_edges = ETRecordGraphBuilder._build_graph_output_incoming_edges(
        output_value,
        output_id="0",
        tensor_producers={},
    )

    assert incoming_edges == []


@pytest.mark.parametrize(
    "output_value",
    [
        None,
        ValueNode(name="output_"),
        ValueNode(name="output", output_shapes=[[1], [2]]),
    ],
)
def test_operator_output_id_falls_back_without_single_output_shape(
    output_value,
):
    producer_operator = OperatorNode(
        name="aten_split_copy_tensor_",
        output_shapes=[[1], [2]],
    )

    producer_output_metadata = builder_module.build_output_metadata(
        producer_operator
    )

    assert (
        ETRecordGraphBuilder._resolve_operator_output_id(
            producer_operator,
            output_value,
        )
        == producer_output_metadata[0].id
    )


@pytest.mark.parametrize(
    ("producer_output_shapes", "graph_output_shape"),
    [
        ([[1], [2]], [3]),  # no match
        ([[1], [1]], [1]),  # ambiguous match
    ],
)
def test_operator_output_id_falls_back_for_non_unique_shape_match(
    producer_output_shapes,
    graph_output_shape,
):
    producer_operator = OperatorNode(
        name="aten_split_copy_tensor_",
        output_shapes=producer_output_shapes,
    )
    output_value = ValueNode(
        name="output",
        output_shapes=[graph_output_shape],
    )

    producer_output_metadata = builder_module.build_output_metadata(
        producer_operator
    )

    assert (
        ETRecordGraphBuilder._resolve_operator_output_id(
            producer_operator,
            output_value,
        )
        == producer_output_metadata[0].id
    )


def test_extract_named_arg_values_ignores_call_function_without_schema():
    exported_node = SimpleNamespace(
        op="call_function",
        target=object(),
        # no schema
    )
    exported_program = SimpleNamespace(
        graph=SimpleNamespace(nodes=[exported_node]),
    )

    assert (
        ETRecordGraphBuilder._extract_named_arg_values(exported_program) == {}
    )
