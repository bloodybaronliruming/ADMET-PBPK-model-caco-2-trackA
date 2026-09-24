"""Audit unchanged official Caco2_Wang structures and saved splits."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem.Scaffolds import MurckoScaffold


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/caco_trackA/raw"
PROCESS = ROOT / "data/caco_trackA/process"
SEEDS = (1, 2, 3, 4, 5)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def clean_set(values: pd.Series) -> set:
    return set(values.dropna()) - {""}


def main() -> None:
    raw_manifest = json.loads((RAW / "dataset_manifest.json").read_text())
    split_manifest = json.loads((PROCESS / "splits_manifest.json").read_text())
    raw = {}
    for part in ("train_val", "test"):
        record = raw_manifest["files"][part]
        path = ROOT / record["path"]
        if digest(path) != record["sha256"]:
            raise ValueError(f"Raw {part} SHA-256 changed")
        raw[part] = pd.read_csv(path)

    structure_records = []
    for part, frame in raw.items():
        row_map = pd.read_csv(PROCESS / "row_ids" / f"{part}.csv")
        if len(row_map) != len(frame) or row_map["source_row"].tolist() != list(range(len(frame))):
            raise ValueError(f"Invalid row identity mapping for {part}")
        for i, smiles in enumerate(frame["Drug"]):
            molecule = Chem.MolFromSmiles(smiles)
            parsed = molecule is not None
            scaffold = (
                MurckoScaffold.MurckoScaffoldSmiles(mol=molecule, includeChirality=False)
                if parsed else ""
            )
            structure_records.append({
                "row_id": row_map.at[i, "row_id"],
                "source_partition": part,
                "source_row": i,
                "Drug_ID": frame.at[i, "Drug_ID"],
                "raw_smiles": smiles,
                "rdkit_parse_ok": parsed,
                "canonical_smiles": Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True) if parsed else "",
                "bemis_murcko_scaffold": scaffold,
                "empty_scaffold": parsed and scaffold == "",
                "fragment_count": len(Chem.GetMolFrags(molecule)) if parsed else 0,
            })
    structure = pd.DataFrame(structure_records)
    if not structure["rdkit_parse_ok"].all():
        raise ValueError("Unparseable source structure; inspect QC before proceeding")
    by_id = structure.set_index("row_id", verify_integrity=True)
    qc_records = []
    for seed in SEEDS:
        groups = {}
        for name in ("train", "valid"):
            path = PROCESS / "splits" / f"seed_{seed}" / f"{name}.csv"
            frame = pd.read_csv(path)
            if len(frame) != split_manifest["splits"][str(seed)][f"{name}_rows"]:
                raise ValueError(f"Seed {seed} {name} row count changed")
            for record in frame.itertuples(index=False):
                original = raw["train_val"].iloc[record.source_row]
                if (record.row_id != f"caco2_wang:train_val:{record.source_row}"
                    or record.source_partition != "train_val" or record.seed != seed
                    or record.split != name or record.Drug_ID != original.Drug_ID
                    or record.Drug != original.Drug or record.Y != original.Y):
                    raise ValueError(f"Seed {seed} {name} record changed")
            groups[name] = by_id.loc[frame["row_id"]]
        groups["test"] = by_id.loc[structure.loc[structure["source_partition"] == "test", "row_id"]]
        if set(groups["train"].index) & set(groups["valid"].index):
            raise ValueError(f"Seed {seed} train/valid row overlap")
        for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
            a, b = groups[left], groups[right]
            qc_records.append({
                "seed": seed,
                "left": left,
                "right": right,
                "left_rows": len(a),
                "right_rows": len(b),
                "row_id_overlap": len(set(a.index) & set(b.index)),
                "drug_id_overlap": len(clean_set(a["Drug_ID"]) & clean_set(b["Drug_ID"])),
                "raw_smiles_overlap": len(clean_set(a["raw_smiles"]) & clean_set(b["raw_smiles"])),
                "canonical_smiles_overlap": len(clean_set(a["canonical_smiles"]) & clean_set(b["canonical_smiles"])),
                "nonempty_scaffold_overlap": len(clean_set(a["bemis_murcko_scaffold"]) & clean_set(b["bemis_murcko_scaffold"])),
                "left_empty_scaffold_rows": int(a["empty_scaffold"].sum()),
                "right_empty_scaffold_rows": int(b["empty_scaffold"].sum()),
            })
    split_qc = pd.DataFrame(qc_records)
    manifest = {
        "dataset": "caco2_wang",
        "dataset_stage": "structural and split QC only; no model feature preprocessing or label transformation",
        "raw_manifest": "data/caco_trackA/raw/dataset_manifest.json",
        "splits_manifest": "data/caco_trackA/process/splits_manifest.json",
        "source_sha256": {part: raw_manifest["files"][part]["sha256"] for part in raw},
        "rdkit_version": rdBase.rdkitVersion,
        "structure_method": {
            "parse": "RDKit Chem.MolFromSmiles on raw Drug",
            "canonical_smiles": "RDKit Chem.MolToSmiles(canonical=True, isomericSmiles=True)",
            "bemis_murcko_scaffold": "RDKit MurckoScaffoldSmiles(includeChirality=False)",
            "empty_scaffold": "empty string means parsed structure with no Murcko scaffold; excluded from overlap count",
            "standardization": "none; no salt removal, charge neutralization, tautomer normalization or stereochemistry removal",
        },
        "rows": {part: len(frame) for part, frame in raw.items()},
        "parse_failures": int((~structure["rdkit_parse_ok"]).sum()),
        "seeds": list(SEEDS),
        "model_feature_preprocessing": "not started; fit train only when implemented",
        "label_transform": "not started; original TDC Y preserved",
        "test_use": "structure and identity integrity QC only; test Y not used for method selection",
    }
    pending = {
        PROCESS / "qc/structure_qc.csv": csv_bytes(structure),
        PROCESS / "qc/split_qc.csv": csv_bytes(split_qc),
        PROCESS / "preprocessing_manifest.json": (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode(),
    }
    for path, content in pending.items():
        if path.exists() and path.read_bytes() != content:
            raise FileExistsError(f"Refusing to overwrite differing QC file: {path}")
    for path, content in pending.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(content)
    print(f"Structure QC: {len(structure)} rows, {manifest['parse_failures']} parse failures")
    print(split_qc.to_string(index=False))
    print(f"Fixed test SHA-256: {digest(RAW / 'caco2_wang_test.csv')}")


if __name__ == "__main__":
    main()
