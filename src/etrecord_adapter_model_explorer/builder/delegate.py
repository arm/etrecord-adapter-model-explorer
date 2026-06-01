# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

from contextlib import suppress
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DelegateGrouping:
    # Example:
    # namespace_by_handle = {
    #     101: "delegates/VgfBackend/partition_9000",
    #     102: "delegates/VgfBackend/partition_9000",
    # }
    # group_attributes = {
    #     "delegates/VgfBackend/partition_9000": {
    #         "Backend": "VgfBackend",
    #         "Instruction Id": "9000",
    #         "Method": "forward",
    #         "Debug Handles": "101, 102",
    #     }
    # }
    namespace_by_handle: dict[int, str]
    group_attributes: dict[str, dict[str, str]]


class ETRecordDelegateBuilder:
    def __init__(self, etrecord: Any) -> None:
        self._etrecord = etrecord

    def build(self) -> DelegateGrouping:
        namespace_by_handle: dict[int, str] = {}
        group_attributes: dict[str, dict[str, str]] = {}

        delegate_map = getattr(self._etrecord, "_delegate_map", None) or {}
        debug_handle_map = (
            getattr(self._etrecord, "_debug_handle_map", None) or {}
        )

        for method_name, method_delegate_map in delegate_map.items():
            if not isinstance(method_delegate_map, dict):
                continue

            method_debug_handle_map = debug_handle_map.get(method_name, {})
            if not isinstance(method_debug_handle_map, dict):
                method_debug_handle_map = {}

            for (
                instruction_id,
                delegate_metadata,
            ) in method_delegate_map.items():
                if not isinstance(delegate_metadata, dict):
                    continue

                backend_name = str(delegate_metadata.get("name") or "Delegate")
                delegate_mapping = delegate_metadata.get("delegate_map") or {}

                # Ignore coarse entries that do not map delegate-internal ids.
                if (
                    not method_debug_handle_map.get(instruction_id)
                    and not delegate_mapping
                ):
                    continue

                namespace = (
                    f"delegates/{backend_name}/partition_{instruction_id}"
                )
                handles = self._collect_partition_handles(
                    method_debug_handle_map.get(instruction_id),
                    delegate_mapping,
                )
                if not handles:
                    continue

                group_attributes[namespace] = {
                    "Backend": backend_name,
                    "Instruction Id": str(instruction_id),
                    "Method": str(method_name),
                    "Debug Handles": ", ".join(
                        str(handle) for handle in handles
                    ),
                }

                for handle in handles:
                    namespace_by_handle[handle] = namespace

        return DelegateGrouping(
            namespace_by_handle=namespace_by_handle,
            group_attributes=group_attributes,
        )

    @staticmethod
    def node_debug_handle(metadata: dict[str, Any]) -> int | None:
        raw_value = metadata.get("debug_handle")
        try:
            return int(raw_value) if raw_value is not None else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _collect_partition_handles(
        instruction_handles: object,
        delegate_mapping: object,
    ) -> list[int]:
        handles: set[int] = set()

        def add_handle(value: Any) -> None:
            with suppress(TypeError, ValueError):
                handles.add(int(value))

        def add_handles(values: object) -> None:
            if isinstance(values, (list, tuple, set)):
                for value in values:
                    add_handle(value)
            else:
                add_handle(values)

        add_handles(instruction_handles)

        if isinstance(delegate_mapping, dict):
            for mapped_handles in delegate_mapping.values():
                add_handles(mapped_handles)

        return sorted(handles)
