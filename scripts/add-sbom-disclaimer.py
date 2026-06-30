#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

"""Add Arm creator metadata and disclaimer text to an SPDX JSON SBOM."""

import argparse
import json
from pathlib import Path

ARM_CREATOR = "Organization: Arm Limited (open-source-office@arm.com)"
SBOM_DISCLAIMER = (
    "THIS SOFTWARE BILL OF MATERIALS (SBOM) IS PROVIDED BY ARM LIMITED "
    "AS IS AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT "
    "LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A "
    "PARTICULAR PURPOSE, AND NONINFRINGEMENT ARE DISCLAIMED. IN NO EVENT "
    "SHALL ARM LIMITED BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, "
    "SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT "
    "LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, "
    "DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY "
    "THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT "
    "(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE "
    "OF THIS SBOM, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add Arm SBOM disclaimer metadata to SPDX JSON"
    )
    parser.add_argument("sbom", type=Path, help="Path to SPDX JSON SBOM")
    return parser.parse_args()


def add_disclaimer(sbom: Path) -> None:
    data = json.loads(sbom.read_text(encoding="utf-8"))
    creation_info = data.setdefault("creationInfo", {})

    creators = creation_info.setdefault("creators", [])
    if ARM_CREATOR not in creators:
        creators.insert(0, ARM_CREATOR)

    existing_comment = creation_info.get("comment")
    comment = SBOM_DISCLAIMER
    if existing_comment and SBOM_DISCLAIMER not in existing_comment:
        comment = f"{existing_comment}\n\n{SBOM_DISCLAIMER}"
    elif existing_comment:
        comment = existing_comment

    creation_info["comment"] = comment
    sbom.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    add_disclaimer(args.sbom)


if __name__ == "__main__":
    main()
