# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

import re
from typing import Any, Iterable, Mapping

from executorch.devtools.debug_format.base_schema import (
    Node,
    OperatorNode,
    ValueNode,
)
from model_explorer import graph_builder as gb

from .util import (
    format_shape,
    format_shapes,
    format_tensor_value,
    safe_str,
)

_SKIPPED_METADATA_KEYS = {"stack_trace", "nn_module_stack"}


def build_node_attrs(
    node: OperatorNode,
    *,
    state_dict: Mapping[str, Any] | None = None,
    placeholder_to_state_name: Mapping[str, str] | None = None,
    named_arg_values: Mapping[str, Any] | None = None,
) -> list[gb.KeyValue]:
    attrs = [
        gb.KeyValue(key="Node Name", value=node.name),
        gb.KeyValue(key="Op Type", value=safe_str(node.op)),
    ]

    attrs.extend(
        _build_value_input_attrs(
            node.inputs or [],
            node.named_args,
            state_dict=state_dict,
            placeholder_to_state_name=placeholder_to_state_name,
        )
    )
    attrs.extend(
        _build_missing_named_arg_attrs(
            node.inputs or [],
            node.named_args,
            named_arg_values=named_arg_values,
        )
    )

    if node.output_shapes:
        attrs.append(
            gb.KeyValue(
                key="Output Shapes",
                value=format_shapes(node.output_shapes),
            )
        )

    metadata = node.metadata or {}
    for key, value in metadata.items():
        if key in _SKIPPED_METADATA_KEYS:
            continue
        if value is None or value == [] or value == {}:
            continue
        attrs.append(gb.KeyValue(key=str(key), value=safe_str(value)))

    return attrs


def _build_value_input_attrs(
    inputs: Iterable[Node],
    named_args: list[str] | None,
    *,
    state_dict: Mapping[str, Any] | None,
    placeholder_to_state_name: Mapping[str, str] | None,
) -> list[gb.KeyValue]:
    attrs: list[gb.KeyValue] = []
    seen_keys: dict[str, int] = {}

    for index, input_node in enumerate(inputs):
        if not isinstance(input_node, ValueNode):
            continue

        key = _display_input_key(
            input_node,
            named_args[index]
            if named_args and index < len(named_args)
            else None,
            placeholder_to_state_name=placeholder_to_state_name,
        )
        value = _display_input_value(
            input_node,
            state_dict=state_dict,
            placeholder_to_state_name=placeholder_to_state_name,
        )
        if key is None or value is None:
            continue

        duplicate_index = seen_keys.get(key, 0)
        seen_keys[key] = duplicate_index + 1
        display_key = (
            key if duplicate_index == 0 else f"{key}_{duplicate_index + 1}"
        )
        attrs.append(gb.KeyValue(key=display_key, value=value))

    return attrs


def _build_missing_named_arg_attrs(
    inputs: Iterable[Node],
    named_args: list[str] | None,
    *,
    named_arg_values: Mapping[str, Any] | None,
) -> list[gb.KeyValue]:
    if not named_args or not named_arg_values:
        return []

    attrs: list[gb.KeyValue] = []
    for named_arg in named_args[len(list(inputs)) :]:
        if named_arg in {"self", "input"}:
            continue
        if named_arg not in named_arg_values:
            continue
        attrs.append(
            gb.KeyValue(
                key=named_arg,
                value=_format_missing_named_arg_value(
                    named_arg_values[named_arg]
                ),
            )
        )
    return attrs


def _display_input_key(
    input_node: ValueNode,
    named_arg: str | None,
    *,
    placeholder_to_state_name: Mapping[str, str] | None,
) -> str | None:
    normalized_name = _normalize_graph_input_name(input_node.name)
    state_name = (placeholder_to_state_name or {}).get(normalized_name)
    if state_name:
        return state_name.rsplit(".", 1)[-1]

    derived_name = _derive_value_node_name(normalized_name)
    if input_node.val is not None:
        return derived_name

    if named_arg not in {None, "self", "input"}:
        if derived_name not in {normalized_name, "p"}:
            return derived_name
        return named_arg

    return None


def _display_input_value(
    input_node: ValueNode,
    *,
    state_dict: Mapping[str, Any] | None,
    placeholder_to_state_name: Mapping[str, str] | None,
) -> str | None:
    normalized_name = _normalize_graph_input_name(input_node.name)
    state_name = (placeholder_to_state_name or {}).get(normalized_name)
    if state_name and state_dict and state_name in state_dict:
        return format_tensor_value(state_dict[state_name])

    if input_node.val is not None:
        return format_shape(input_node.val)

    parts = []
    if input_node.output_shapes:
        parts.append(f"shape={format_shapes(input_node.output_shapes)}")
    if input_node.dtype:
        parts.append(f"dtype={safe_str(input_node.dtype)}")
    if parts:
        return ", ".join(parts)
    return None


def _normalize_graph_input_name(name: str) -> str:
    return name[:-1] if name.endswith("_") else name


def _derive_value_node_name(name: str) -> str:
    return re.sub(r"_\d+$", "", name)


def _format_missing_named_arg_value(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, (list, tuple)):
        return format_shape(value)
    return safe_str(value)


def build_output_metadata(
    node: Node, id_prefix: str = "0", *, aggregate_shapes: bool = False
) -> list[gb.MetadataItem]:
    shapes = getattr(node, "output_shapes", None) or []
    if not shapes:
        return []

    metadata = getattr(node, "metadata", None) or {}
    debug_handle = metadata.get("debug_handle")
    dtype = getattr(node, "dtype", "")

    if aggregate_shapes:
        attrs = [gb.KeyValue(key="Shape", value=format_shapes(shapes))]
        if dtype:
            attrs.append(gb.KeyValue(key="DType", value=safe_str(dtype)))
        if debug_handle is not None:
            attrs.append(
                gb.KeyValue(key="Debug Handle", value=safe_str(debug_handle))
            )
        return [gb.MetadataItem(id=id_prefix, attrs=attrs)]

    items: list[gb.MetadataItem] = []
    for index, shape in enumerate(shapes):
        output_id = id_prefix if len(shapes) == 1 else f"{id_prefix}:{index}"
        attrs = [gb.KeyValue(key="Shape", value=format_shape(shape))]
        if dtype:
            attrs.append(gb.KeyValue(key="DType", value=safe_str(dtype)))
        if debug_handle is not None:
            attrs.append(
                gb.KeyValue(key="Debug Handle", value=safe_str(debug_handle))
            )
        items.append(gb.MetadataItem(id=output_id, attrs=attrs))

    return items


def build_input_metadata(
    inputs: Iterable[Node],
    named_args: list[str] | None,
) -> list[gb.MetadataItem]:
    metadata_items: list[gb.MetadataItem] = []
    for index, input_node in enumerate(inputs):
        attrs = [gb.KeyValue(key="Source Node", value=input_node.name)]

        if named_args and index < len(named_args):
            attrs.append(
                gb.KeyValue(key="Arg Name", value=safe_str(named_args[index]))
            )

        input_shapes = getattr(input_node, "output_shapes", None) or []
        if input_shapes:
            attrs.append(
                gb.KeyValue(key="Shape", value=format_shapes(input_shapes))
            )

        if isinstance(input_node, ValueNode):
            if input_node.dtype:
                attrs.append(
                    gb.KeyValue(key="DType", value=safe_str(input_node.dtype))
                )
            if input_node.val is not None:
                attrs.append(
                    gb.KeyValue(
                        key="Const Value", value=safe_str(input_node.val)
                    )
                )

        metadata = getattr(input_node, "metadata", None) or {}
        debug_handle = metadata.get("debug_handle")
        if debug_handle is not None:
            attrs.append(
                gb.KeyValue(key="Debug Handle", value=safe_str(debug_handle))
            )

        metadata_items.append(gb.MetadataItem(id=str(index), attrs=attrs))

    return metadata_items
