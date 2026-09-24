"""Freeze the RDKit 2023.09.6 two-dimensional descriptor list before calculation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from rdkit import rdBase
from rdkit.Chem import Descriptors, Descriptors3D


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "data/caco_trackA/process/features/rdkit2d"
EXPECTED_RDKIT = "2023.09.6"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stable_json(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()


def callable_record(name: str, function: object, source: str, order: int) -> dict:
    if not callable(function):
        raise ValueError(f"{source} entry is not callable: {name}")
    return {
        "candidate_order": order,
        "name": name,
        "source_registry": source,
        "calculation_function": f"rdkit.Chem.Descriptors.{name}" if source == "Descriptors.descList"
        else f"rdkit.Chem.Descriptors3D.{name}",
        "implementation_module": getattr(function, "__module__", None),
        "implementation_name": getattr(function, "__qualname__", getattr(function, "__name__", None)),
    }


def write_frozen(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(f"Frozen descriptor file differs: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def main() -> None:
    if rdBase.rdkitVersion != EXPECTED_RDKIT:
        raise RuntimeError(
            f"RDKit version changed: {rdBase.rdkitVersion}; expected {EXPECTED_RDKIT}"
        )

    two_d = list(Descriptors.descList)
    three_d = list(Descriptors3D.descList)
    two_d_names = [name for name, _ in two_d]
    three_d_names = [name for name, _ in three_d]
    if (
        len(two_d) != 210
        or len(three_d) != 11
        or len(set(two_d_names)) != len(two_d_names)
        or len(set(three_d_names)) != len(three_d_names)
        or set(two_d_names) & set(three_d_names)
    ):
        raise ValueError("Unexpected RDKit descriptor registry composition")

    candidate_list = [
        callable_record(name, function, "Descriptors.descList", i)
        for i, (name, function) in enumerate(two_d, start=1)
    ] + [
        callable_record(name, function, "Descriptors3D.descList", len(two_d) + i)
        for i, (name, function) in enumerate(three_d, start=1)
    ]
    selected = [
        {"feature_order": i, **{key: value for key, value in item.items() if key != "candidate_order"}}
        for i, item in enumerate(candidate_list[: len(two_d)], start=1)
    ]
    excluded = [
        {**item, "exclusion_reason": "3D/conformer-dependent descriptor; excluded by the predeclared 2D-only rule"}
        for item in candidate_list[len(two_d) :]
    ]
    descriptor_names = [item["name"] for item in selected]
    manifest = {
        "dataset": "caco2_wang",
        "representation": "RDKit2D",
        "rdkit_version": rdBase.rdkitVersion,
        "definition_sources": [
            "descriptors.md section 3.1",
            "endpoint_1_caco_trackA_baseline.md section 6",
            "RDKit rdkit.Chem.Descriptors.descList and rdkit.Chem.Descriptors3D.descList",
        ],
        "selection_rule": (
            "Before calculating any dataset features or inspecting labels, take the RDKit "
            "Descriptors.descList entries in registry order as deterministic 2D molecular "
            "descriptors. Exclude every Descriptors3D.descList entry because it requires "
            "a 3D conformer. No filtering by dataset values, validation scores or test results."
        ),
        "candidate_count": len(candidate_list),
        "selected_count": len(selected),
        "excluded_count": len(excluded),
        "candidate_list": candidate_list,
        "selected_descriptors": selected,
        "descriptor_names_in_order": descriptor_names,
        "descriptor_names_sha256": digest("\n".join(descriptor_names).encode()),
        "excluded_descriptors": excluded,
        "feature_calculation_status": "not started; manifest frozen before matrix construction",
        "freeze_policy": "refuse to overwrite a differing manifest or SHA-256 sidecar",
    }
    manifest_path = OUTPUT / "rdkit2d_descriptor_manifest.json"
    content = stable_json(manifest)
    write_frozen(manifest_path, content)
    checksum_path = OUTPUT / "rdkit2d_descriptor_manifest.sha256"
    write_frozen(checksum_path, (digest(content) + "  rdkit2d_descriptor_manifest.json\n").encode())
    print(f"RDKit {rdBase.rdkitVersion}: {len(candidate_list)} candidates; "
          f"{len(selected)} retained 2D descriptors; {len(excluded)} excluded 3D descriptors")
    print(f"Frozen manifest: {manifest_path.relative_to(ROOT)}")
    print(f"Manifest SHA-256: {digest(content)}")


if __name__ == "__main__":
    main()
