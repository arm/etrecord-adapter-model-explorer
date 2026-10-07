# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

from etrecord_adapter_model_explorer.builder.builder import (
    BuildContext,
)


def empty_build_context() -> BuildContext:
    return BuildContext(
        delegate_namespace_by_handle={},
        delegate_group_attributes={},
        parameter_state_dict={},
        placeholder_to_state_name={},
        named_arg_values_by_node={},
    )


def key_value_attrs(items) -> dict[str, str]:
    return {item.key: item.value for item in items}
