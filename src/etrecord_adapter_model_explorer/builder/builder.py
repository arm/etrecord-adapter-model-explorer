# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Dict

from executorch.devtools.debug_format.base_schema import (
    Node,
    OperatorGraph,
    OperatorNode,
    ValueNode,
)
from executorch.devtools.etrecord import parse_etrecord
from executorch.devtools.inspector._inspector_utils import (
    EDGE_DIALECT_GRAPH_KEY,
    gen_graphs_from_etrecord,
)
from model_explorer import graph_builder as gb

from ..constants import (
    ADAPTER_NAME,
    GRAPH_INPUT_ANNOTATION,
    GRAPH_OUTPUT_ANNOTATION,
)
from .delegate import ETRecordDelegateBuilder
from .metadata import (
    build_input_metadata,
    build_node_attrs,
    build_output_metadata,
)
from .util import module_namespace, node_label, safe_str

_ETRECORD_LOAD_LOCK = Lock()


@dataclass(frozen=True)
class BuildContext:
    delegate_namespace_by_handle: dict[int, str]
    delegate_group_attributes: dict[str, dict[str, str]]
    parameter_state_dict: dict[str, Any]
    placeholder_to_state_name: dict[str, str]
    named_arg_values_by_node: dict[str, dict[str, Any]]


class ETRecordGraphBuilder:
    """Builds Model Explorer graphs from ExecuTorch ETRecord file."""

    def __init__(self, source_path: Path, settings: Dict) -> None:
        del settings
        self.source_path = self._resolve_etrecord_path(source_path)
        self.model_name = self.source_path.name

    def build(self) -> list[gb.GraphCollection]:
        etrecord, operator_graphs, _edge_program_key = (
            self._load_graph_contexts()
        )
        delegate_grouping = ETRecordDelegateBuilder(etrecord).build()
        graphs = [
            self._build_model_explorer_graph(
                self.source_path,
                operator_graph,
                self._build_context(
                    self._find_exported_program(etrecord, graph_key),
                    delegate_grouping=(
                        delegate_grouping
                        if self._graph_uses_delegate_handles(
                            operator_graph,
                            delegate_grouping.namespace_by_handle,
                        )
                        else None
                    ),
                ),
                graph_id=graph_key,
            )
            for graph_key, operator_graph in operator_graphs.items()
        ]
        return [gb.GraphCollection(label=self.model_name, graphs=graphs)]

    @property
    def graph_collections(self) -> list[gb.GraphCollection]:
        return self.build()

    def _load_graph_contexts(
        self,
    ) -> tuple[object, Mapping[str, OperatorGraph], str]:
        # ExecuTorch deserialization uses process-global state guarded by an
        # assertion, so concurrent Model Explorer loads can trip it.
        with _ETRECORD_LOAD_LOCK:
            etrecord = parse_etrecord(str(self.source_path))
            operator_graphs = self._gen_operator_graphs(etrecord)
        if not operator_graphs:
            raise ValueError(
                "This ETRecord does not contain any graph artifacts."
            )

        edge_program_key = self._find_edge_program_key(operator_graphs)
        if edge_program_key is None:
            raise ValueError(
                "This ETRecord contains graph artifacts, but none were "
                "identified as an edge dialect graph."
            )

        if self._find_exported_program(etrecord, edge_program_key) is None:
            raise ValueError(
                "This ETRecord contains an edge dialect graph, but the matching "
                "edge dialect exported program could not be found."
            )

        return etrecord, operator_graphs, edge_program_key

    @staticmethod
    def _gen_operator_graphs(etrecord) -> Mapping[str, OperatorGraph]:
        edge_dialect_program = etrecord.edge_dialect_program
        if not isinstance(edge_dialect_program, dict):
            return gen_graphs_from_etrecord(etrecord)

        # ExecuTorch can parse multi-method ETRecords as a dict of exported
        # programs, but gen_graphs_from_etrecord still assumes a single program
        # and accesses edge_dialect_program.graph_module. Feed the upstream
        # helper single-method ETRecord views until it handles this shape.
        graph_map_etrecord = copy(etrecord)
        graph_map_etrecord.edge_dialect_program = None
        graph_map_operator_graphs = gen_graphs_from_etrecord(
            graph_map_etrecord
        )
        operator_graphs = {
            key: graph_map_operator_graphs[key]
            for key in sorted(graph_map_operator_graphs)
        }

        for method_name, exported_program in sorted(
            edge_dialect_program.items(),
            key=lambda item: (item[0] != "forward", item[0]),
        ):
            method_etrecord = copy(etrecord)
            method_etrecord.graph_map = None
            method_etrecord.edge_dialect_program = exported_program
            method_graphs = gen_graphs_from_etrecord(method_etrecord)

            graph_key = (
                EDGE_DIALECT_GRAPH_KEY
                if method_name == "forward"
                else f"edge_dialect_exported_program/{method_name}"
            )
            operator_graphs[graph_key] = method_graphs[EDGE_DIALECT_GRAPH_KEY]

        return operator_graphs

    @staticmethod
    def _find_edge_program_key(
        operator_graphs: Mapping[str, OperatorGraph],
    ) -> str | None:
        if EDGE_DIALECT_GRAPH_KEY in operator_graphs:
            return EDGE_DIALECT_GRAPH_KEY

        for graph_key in operator_graphs:
            if graph_key.startswith("edge_dialect_exported_program"):
                return graph_key
        return None

    @staticmethod
    def _find_exported_program(etrecord, graph_key: str):
        if graph_key == "exported_program":
            return etrecord.exported_program

        edge_dialect_program = etrecord.edge_dialect_program
        if isinstance(edge_dialect_program, dict):
            method_name = ETRecordGraphBuilder._edge_method_name(graph_key)
            if method_name in edge_dialect_program:
                return edge_dialect_program[method_name]
        elif edge_dialect_program is not None and (
            graph_key == EDGE_DIALECT_GRAPH_KEY
            or graph_key.startswith("edge_dialect_exported_program")
        ):
            return edge_dialect_program

        graph_map = etrecord.graph_map or {}
        if graph_key in graph_map:
            return graph_map[graph_key]

        if graph_key == EDGE_DIALECT_GRAPH_KEY:
            return graph_map.get("edge_dialect_exported_program")

        return None

    @staticmethod
    def _edge_method_name(graph_key: str) -> str:
        if graph_key.startswith("edge_dialect_exported_program/"):
            return graph_key.removeprefix("edge_dialect_exported_program/")
        return "forward"

    def _build_context(
        self,
        exported_program,
        *,
        delegate_grouping,
    ) -> BuildContext:
        if exported_program is None:
            return BuildContext({}, {}, {}, {}, {})

        graph_signature = exported_program.graph_signature
        return BuildContext(
            delegate_namespace_by_handle=(
                delegate_grouping.namespace_by_handle
                if delegate_grouping is not None
                else {}
            ),
            delegate_group_attributes=(
                delegate_grouping.group_attributes
                if delegate_grouping is not None
                else {}
            ),
            parameter_state_dict=dict(exported_program.state_dict),
            placeholder_to_state_name={
                **graph_signature.inputs_to_parameters,
                **graph_signature.inputs_to_buffers,
                **graph_signature.inputs_to_lifted_tensor_constants,
            },
            named_arg_values_by_node=self._extract_named_arg_values(
                exported_program
            ),
        )

    @classmethod
    def _graph_uses_delegate_handles(
        cls,
        operator_graph: OperatorGraph,
        namespace_by_handle: Mapping[int, str],
    ) -> bool:
        if not namespace_by_handle:
            return False

        forward_graph = cls._find_subgraph(operator_graph, "forward")
        if forward_graph is None:
            return False

        for element in forward_graph.elements:
            if not isinstance(element, OperatorNode):
                continue
            debug_handle = ETRecordDelegateBuilder.node_debug_handle(
                element.metadata or {}
            )
            if debug_handle in namespace_by_handle:
                return True
        return False

    @classmethod
    def _build_model_explorer_graph(
        cls,
        source_path: Path,
        edge_graph: OperatorGraph,
        context: BuildContext,
        graph_id: str = "forward",
    ) -> gb.Graph:
        inputs_graph = cls._find_subgraph(edge_graph, "inputs")
        forward_graph = cls._find_subgraph(edge_graph, "forward")
        outputs_graph = cls._find_subgraph(edge_graph, "outputs")

        if (
            inputs_graph is None
            or forward_graph is None
            or outputs_graph is None
        ):
            raise ValueError(
                "Unexpected ETRecord edge-dialect graph format: missing one of "
                "'inputs', 'forward', or 'outputs' sections."
            )

        tensor_producers: dict[str, tuple[str, str]] = {}
        nodes: list[gb.GraphNode] = []

        nodes.append(
            cls._build_graph_inputs_node(inputs_graph, tensor_producers)
        )
        nodes.extend(
            cls._build_operator_nodes(
                forward_graph,
                context=context,
                tensor_producers=tensor_producers,
            )
        )
        nodes.append(
            cls._build_graph_outputs_node(outputs_graph, tensor_producers)
        )

        return gb.Graph(
            id=cls._model_explorer_graph_id(graph_id),
            nodes=nodes,
            groupNodeAttributes={
                "": {
                    "Adapter": ADAPTER_NAME,
                    "ETRecord Path": str(source_path),
                    "ETRecord Graph": graph_id,
                },
                **context.delegate_group_attributes,
            },
            groupNodeConfigs=(
                [
                    gb.GroupNodeConfig(
                        namespaceRegex=r"delegates(/.*)?",
                        layoutDirection=0,
                    )
                ]
                if context.delegate_group_attributes
                else None
            ),
        )

    @staticmethod
    def _build_graph_inputs_node(
        inputs_graph: OperatorGraph,
        tensor_producers: dict[str, tuple[str, str]],
    ) -> gb.GraphNode:
        output_metadata: list[gb.MetadataItem] = []
        for value_node in inputs_graph.elements:
            if not isinstance(value_node, ValueNode):
                continue

            # Only treat true runtime inputs as coming from the synthetic Inputs node.
            # Parameters/buffers live in the inputs subgraph too, but they should not
            # appear as graph edges from Inputs.
            if value_node.name.startswith(("p_", "b_")):
                continue

            tensor_producers[value_node.name] = (
                GRAPH_INPUT_ANNOTATION,
                value_node.name,
            )
            output_metadata.extend(
                build_output_metadata(value_node, id_prefix=value_node.name)
            )

        return gb.GraphNode(
            id=GRAPH_INPUT_ANNOTATION,
            label="Graph Inputs",
            namespace=GRAPH_INPUT_ANNOTATION,
            outputsMetadata=output_metadata,
        )

    @classmethod
    def _build_operator_nodes(
        cls,
        forward_graph: OperatorGraph,
        context: BuildContext,
        tensor_producers: dict[str, tuple[str, str]],
    ) -> list[gb.GraphNode]:
        operator_nodes: list[gb.GraphNode] = []

        for op_node in forward_graph.elements:
            if not isinstance(op_node, OperatorNode):
                continue

            incoming_edges = []
            for input_index, input_node in enumerate(op_node.inputs or []):
                producer = cls._resolve_tensor_producer(
                    input_node,
                    tensor_producers=tensor_producers,
                )
                if producer is None:
                    continue
                source_node_id, source_output_id = producer
                incoming_edges.append(
                    gb.IncomingEdge(
                        sourceNodeId=source_node_id,
                        sourceNodeOutputId=source_output_id,
                        targetNodeInputId=str(input_index),
                    )
                )

            metadata = op_node.metadata or {}
            debug_handle = ETRecordDelegateBuilder.node_debug_handle(metadata)
            namespace = (
                context.delegate_namespace_by_handle.get(debug_handle)
                if debug_handle is not None
                else None
            )
            node = gb.GraphNode(
                id=op_node.name,
                label=node_label(op_node),
                namespace=namespace
                or module_namespace(metadata.get("nn_module_stack")),
                attrs=build_node_attrs(
                    op_node,
                    state_dict=context.parameter_state_dict,
                    placeholder_to_state_name=context.placeholder_to_state_name,
                    named_arg_values=context.named_arg_values_by_node.get(
                        cls._normalize_debug_name(op_node.name)
                    ),
                ),
                incomingEdges=incoming_edges,
                inputsMetadata=build_input_metadata(
                    op_node.inputs or [],
                    op_node.named_args,
                ),
                outputsMetadata=build_output_metadata(op_node),
            )

            tensor_producers[op_node.name] = (op_node.name, "0")
            operator_nodes.append(node)

        return operator_nodes

    @classmethod
    def _build_graph_outputs_node(
        cls,
        outputs_graph: OperatorGraph,
        tensor_producers: dict[str, tuple[str, str]],
    ) -> gb.GraphNode:
        inputs_metadata: list[gb.MetadataItem] = []
        incoming_edges: list[gb.IncomingEdge] = []

        output_values = [
            value_node
            for value_node in outputs_graph.elements
            if isinstance(value_node, ValueNode)
        ]

        for index, output_value in enumerate(output_values):
            output_id = str(index)
            metadata_item = cls._build_graph_output_metadata_item(
                output_value, output_id
            )
            if metadata_item is not None:
                inputs_metadata.append(metadata_item)
            incoming_edges.extend(
                cls._build_graph_output_incoming_edges(
                    output_value,
                    output_id=output_id,
                    tensor_producers=tensor_producers,
                )
            )

        return gb.GraphNode(
            id=GRAPH_OUTPUT_ANNOTATION,
            label="Graph Outputs",
            namespace=GRAPH_OUTPUT_ANNOTATION,
            incomingEdges=incoming_edges,
            inputsMetadata=inputs_metadata,
        )

    @classmethod
    def _build_graph_output_metadata_item(
        cls, output_value: ValueNode, output_id: str
    ) -> gb.MetadataItem | None:
        source = cls._graph_output_source(output_value)
        metadata_value = cls._graph_output_metadata_value(
            output_value,
            source,
        )
        metadata_items = build_output_metadata(
            metadata_value,
            id_prefix=output_id,
            aggregate_shapes=True,
        )
        if not metadata_items:
            return None

        metadata_item = metadata_items[0]
        if source is not None:
            metadata_item.attrs.insert(
                0,
                gb.KeyValue(key="Source Node", value=source.name),
            )
        if metadata_value.val is not None:
            metadata_item.attrs.append(
                gb.KeyValue(
                    key="Const Value",
                    value=safe_str(metadata_value.val),
                )
            )
        return metadata_item

    @classmethod
    def _build_graph_output_incoming_edges(
        cls,
        output_value: ValueNode,
        *,
        output_id: str,
        tensor_producers: dict[str, tuple[str, str]],
    ) -> list[gb.IncomingEdge]:
        incoming_edges: list[gb.IncomingEdge] = []
        for source in output_value.inputs or []:
            producer = cls._resolve_tensor_producer(
                source,
                tensor_producers=tensor_producers,
                output_value=output_value,
            )
            if producer is None:
                continue

            source_node_id, source_output_id = producer
            incoming_edges.append(
                gb.IncomingEdge(
                    sourceNodeId=source_node_id,
                    sourceNodeOutputId=source_output_id,
                    targetNodeInputId=output_id,
                )
            )
        return incoming_edges

    @staticmethod
    def _graph_output_source(output_value: ValueNode) -> Node | None:
        source_nodes = output_value.inputs or []
        return source_nodes[0] if len(source_nodes) == 1 else None

    @staticmethod
    def _graph_output_metadata_value(
        output_value: ValueNode,
        source: Node | None,
    ) -> ValueNode:
        metadata_value = ValueNode(
            name=output_value.name,
            output_shapes=(
                output_value.output_shapes
                or getattr(source, "output_shapes", None)
            ),
            metadata=(
                getattr(source, "metadata", None)
                if source is not None
                else None
            )
            or output_value.metadata,
            dtype="",
            val=None,
        )
        if isinstance(source, ValueNode):
            metadata_value.dtype = source.dtype or output_value.dtype
            metadata_value.val = (
                source.val if source.val is not None else output_value.val
            )
        elif source is None:
            metadata_value.dtype = output_value.dtype
            metadata_value.val = output_value.val
        return metadata_value

    @classmethod
    def _resolve_tensor_producer(
        cls,
        source: Node,
        *,
        tensor_producers: dict[str, tuple[str, str]],
        output_value: ValueNode | None = None,
    ) -> tuple[str, str] | None:
        if isinstance(source, OperatorNode):
            return (
                source.name,
                cls._resolve_operator_output_id(source, output_value),
            )

        return tensor_producers.get(source.name)

    @staticmethod
    def _resolve_operator_output_id(
        source: OperatorNode,
        output_value: ValueNode | None = None,
    ) -> str:
        output_metadata = build_output_metadata(source)
        if len(output_metadata) <= 1:
            return "0"

        output_shapes = source.output_shapes or []
        if (
            output_value is None
            or output_value.output_shapes is None
            or len(output_value.output_shapes) != 1
        ):
            return output_metadata[0].id

        matching_indexes = [
            index
            for index, source_shape in enumerate(output_shapes)
            if source_shape == output_value.output_shapes[0]
        ]
        if len(matching_indexes) != 1:
            return output_metadata[0].id

        return output_metadata[matching_indexes[0]].id

    @staticmethod
    def _resolve_etrecord_path(source_path: Path) -> Path:
        if not source_path.exists():
            raise FileNotFoundError(
                f"Source file {source_path} does not exist."
            )

        if source_path.suffix.lower() != ".etrecord":
            raise ValueError(
                f"Unsupported file type {source_path.suffix!r}. Expected .etrecord."
            )

        return source_path

    @staticmethod
    def _model_explorer_graph_id(graph_key: str) -> str:
        if graph_key == EDGE_DIALECT_GRAPH_KEY:
            return "forward"
        if graph_key.startswith("edge_dialect_exported_program/"):
            return graph_key.removeprefix("edge_dialect_exported_program/")
        return graph_key

    @staticmethod
    def _find_subgraph(
        op_graph: OperatorGraph,
        graph_name: str,
    ) -> OperatorGraph | None:
        for element in op_graph.elements:
            if (
                isinstance(element, OperatorGraph)
                and element.graph_name == graph_name
            ):
                return element
        return None

    @staticmethod
    def _normalize_debug_name(name: str) -> str:
        return name[:-1] if name.endswith("_") else name

    @classmethod
    def _extract_named_arg_values(
        cls, exported_program
    ) -> dict[str, dict[str, Any]]:
        values_by_node: dict[str, dict[str, Any]] = {}
        for node in exported_program.graph.nodes:
            if node.op != "call_function":
                continue

            schema = getattr(node.target, "_schema", None)
            if schema is None:
                continue

            node_values: dict[str, Any] = {}
            schema_args = list(schema.arguments)
            positional_count = len(node.args)
            for arg in schema_args[positional_count:]:
                node_values[arg.name] = node.kwargs.get(
                    arg.name, arg.default_value
                )

            if node_values:
                values_by_node[cls._normalize_debug_name(node.name)] = (
                    node_values
                )

        return values_by_node
