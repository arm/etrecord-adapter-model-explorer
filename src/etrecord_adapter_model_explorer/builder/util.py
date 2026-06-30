# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Iterable

from executorch.devtools.debug_format.base_schema import OperatorNode


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    return str(value)


def format_shape(shape: Any) -> str:
    if shape is None:
        return ""
    if isinstance(shape, Iterable) and not isinstance(shape, (str, bytes)):
        return "[" + ", ".join(safe_str(dim) for dim in shape) + "]"
    return safe_str(shape)


def format_shapes(shapes: Any) -> str:
    if not shapes:
        return ""
    return ", ".join(format_shape(shape) for shape in shapes)


def format_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def format_sequence(values: Iterable[Any], limit: int = 6) -> str:
    items = [safe_str(value) for value in values]
    if len(items) <= limit:
        return "[" + ", ".join(items) + "]"
    head = ", ".join(items[:limit])
    return f"[{head}, ...] ({len(items)} total)"


def format_op_label(op: Any) -> str:
    op_str = safe_str(op)
    if not op_str:
        return ""

    if "::" in op_str:
        namespace, _, remainder = op_str.partition("::")
        op_name = remainder.split(".", 1)[0]
        if namespace and op_name:
            return f"{namespace}::{op_name}"
        return op_str

    pieces = op_str.split(".")
    if len(pieces) >= 2:
        return f"{pieces[0]}::{pieces[1]}"

    return op_str


def format_tensor_value(value: Any, limit: int = 6) -> str:
    tensor = value.detach() if hasattr(value, "detach") else value
    shape = getattr(tensor, "shape", None)
    dtype = getattr(tensor, "dtype", None)

    flat = tensor.flatten() if hasattr(tensor, "flatten") else None
    total = (
        int(flat.numel()) if flat is not None and hasattr(flat, "numel") else 0
    )
    if flat is not None and total > 0:
        head = flat[:limit]
        head_values = head.tolist() if hasattr(head, "tolist") else list(head)
        sample = "[" + ", ".join(
            _format_tensor_sample_item(item) for item in head_values
        )
        if total > limit:
            sample += f", ...] ({total} total)"
        else:
            sample += "]"
    else:
        sample = safe_str(tensor)

    parts = []
    if shape is not None:
        parts.append(f"shape={format_shape(list(shape))}")
    if dtype is not None:
        parts.append(f"dtype={safe_str(dtype)}")
    parts.append(f"values={sample}")
    return ", ".join(parts)


def _format_tensor_sample_item(item: Any) -> str:
    if isinstance(item, bool):
        return safe_str(item)
    if isinstance(item, (int, float)):
        return format_float(float(item))
    return safe_str(item)


def node_label(op_node: OperatorNode) -> str:
    metadata = op_node.metadata or {}
    module_stack = metadata.get("nn_module_stack") or {}
    if module_stack:
        for module_name, module_type in reversed(list(module_stack.values())):
            if module_name and module_type:
                return module_type.rsplit(".", 1)[-1]

    return format_op_label(op_node.op) or op_node.name


def module_namespace(module_stack: dict[str, tuple[str, str]] | None) -> str:
    if not module_stack:
        return ""

    module_names = []
    for module_name, _ in module_stack.values():
        if module_name:
            module_names.append(module_name)

    return "/".join(module_names)


def interpolate_heat_color(
    value: float,
    min_value: float,
    max_value: float,
    low_rgb: tuple[int, int, int] = (232, 245, 233),
    high_rgb: tuple[int, int, int] = (255, 205, 210),
) -> str:
    if max_value <= min_value:
        ratio = 0.5
    else:
        ratio = (value - min_value) / (max_value - min_value)

    ratio = min(1.0, max(0.0, ratio))
    rgb = tuple(
        round(low + (high - low) * ratio)
        for low, high in zip(low_rgb, high_rgb, strict=True)
    )
    return "#" + "".join(f"{channel:02x}" for channel in rgb)
