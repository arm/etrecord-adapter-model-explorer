# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from etrecord_adapter_model_explorer.builder.delegate import (
    ETRecordDelegateBuilder,
)


def test_collect_partition_handles_deduplicates_and_filters_invalid_values():
    handles = ETRecordDelegateBuilder._collect_partition_handles(
        instruction_handles=[100, "101", "bad", 100],
        delegate_mapping={
            "delegate_0": [101, "102", object()],
            "delegate_1": 103,
            "delegate_2": ["104", None],
            "delegate_3": "skip",
        },
    )

    assert handles == [100, 101, 102, 103, 104]


def test_node_debug_handle_parses_valid_and_invalid_values():
    assert (
        ETRecordDelegateBuilder.node_debug_handle({"debug_handle": "7"}) == 7
    )
    assert (
        ETRecordDelegateBuilder.node_debug_handle({"debug_handle": "bad"})
        is None
    )
    assert ETRecordDelegateBuilder.node_debug_handle({}) is None


def test_delegate_builder_formats_namespaces_and_skips_invalid_entries():
    etrecord = SimpleNamespace(
        _delegate_map={
            "forward": {
                9000: {
                    "name": "TestBackend",
                    "delegate_map": {
                        "delegate_0": [101, "102", "bad"],
                        "delegate_1": 102,
                    },
                },
                9001: {
                    "delegate_map": {
                        "delegate_0": [201],
                    }
                },
                9002: {"name": "SkipNoMap"},
                9003: {
                    "name": "SkipInvalidHandles",
                    "delegate_map": {"d": ["bad"]},
                },
                "malformed": "not-a-dict",
            },
            "noise": "not-a-dict",
        }
    )

    grouping = ETRecordDelegateBuilder(etrecord).build()

    assert grouping.namespace_by_handle == {
        101: "delegates/TestBackend/partition_9000",
        102: "delegates/TestBackend/partition_9000",
        201: "delegates/Delegate/partition_9001",
    }
    assert grouping.group_attributes == {
        "delegates/TestBackend/partition_9000": {
            "Backend": "TestBackend",
            "Instruction Id": "9000",
            "Method": "forward",
            "Debug Handles": "101, 102",
        },
        "delegates/Delegate/partition_9001": {
            "Backend": "Delegate",
            "Instruction Id": "9001",
            "Method": "forward",
            "Debug Handles": "201",
        },
    }


def test_delegate_builder_uses_debug_handle_map_when_delegate_map_is_empty():
    etrecord = SimpleNamespace(
        _debug_handle_map={
            "forward": {
                9000: [101, 102],
            }
        },
        _delegate_map={
            "forward": {
                9000: {
                    "name": "TestBackend",
                    "delegate_map": {},
                }
            }
        },
    )

    grouping = ETRecordDelegateBuilder(etrecord).build()

    assert grouping.namespace_by_handle == {
        101: "delegates/TestBackend/partition_9000",
        102: "delegates/TestBackend/partition_9000",
    }
    assert grouping.group_attributes == {
        "delegates/TestBackend/partition_9000": {
            "Backend": "TestBackend",
            "Instruction Id": "9000",
            "Method": "forward",
            "Debug Handles": "101, 102",
        }
    }
