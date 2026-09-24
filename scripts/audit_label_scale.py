"""Compare cited primary measurements with Wang SI1 and TDC train_val only."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

from rdkit import Chem


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--si", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    root = args.root
    train_path = root / "data/caco_trackA/raw/caco2_wang_train_val.csv"
    pairs_path = root / "references/caco_label_scale_pairs.csv"
    expected_si_hash = "8fda0f4552645565eb890c064b94934021977cc78df6e62ae4e6fe503ef8ba3b"
    if digest(args.si) != expected_si_hash:
        raise ValueError("SI workbook differs from the audited source version")
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(args.si) as z:
        strings = ["".join(e.itertext()) for e in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", ns)]
        si = {}
        for row in ET.fromstring(z.read("xl/worksheets/sheet1.xml")).findall("m:sheetData/m:row", ns):
            cells = {}
            for cell in row.findall("m:c", ns):
                value = cell.find("m:v", ns)
                if value is not None:
                    col = "".join(c for c in cell.attrib["r"] if c.isalpha())
                    cells[col] = strings[int(value.text)] if cell.attrib.get("t") == "s" else value.text
            si[int(row.attrib["r"])] = cells
    with train_path.open(newline="") as f:
        train = list(csv.DictReader(f))
    with pairs_path.open(newline="") as f:
        pairs = list(csv.DictReader(f))
    report = []
    for pair in pairs:
        cells = si[int(pair["si1_excel_row"])]
        if cells["B"] != pair["tdc_name"]:
            raise ValueError("SI row/name mismatch")
        candidates = [(i, r) for i, r in enumerate(train) if r["Drug_ID"] == pair["tdc_name"]]
        if len(candidates) != 1:
            raise ValueError("Expected a single train_val record for this source pair")
        index, row = candidates[0]
        y = float(row["Y"])
        if abs(y - float(cells["E"])) > 1e-6:
            raise ValueError("TDC and SI labels disagree")
        canon = lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s), isomericSmiles=True)
        if canon(row["Drug"]) != canon(cells["C"]):
            raise ValueError("TDC and SI molecular structures disagree")
        p = float(pair["papp_table_value"]) * 1e-6
        proposed = math.log10(p)
        report.append({**pair, "row_id": f"caco2_wang:train_val:{index}",
                       "tdc_y": y, "si1_y": float(cells["E"]),
                       "papp_cm_s": p, "log10_cm_s": proposed,
                       "difference_y_minus_log10_cm_s": y - proposed,
                       "ln_cm_s": math.log(p), "log10_micro_cm_s": math.log10(p / 1e-6),
                       "roundtrip_relative_error": abs(10**proposed / p - 1),
                       "half_mean_with_2decimal_log_hypothesis": (proposed + round(proposed, 2)) / 2})
    result = {"scope": "source-scale audit; train_val only; no labels modified",
              "source_dois": sorted({pair["source_doi"] for pair in pairs}), "source_locations": sorted({pair["source_location"] for pair in pairs}),
              "input_sha256": {"si": digest(args.si), "train_val": digest(train_path), "pairs": digest(pairs_path)},
              "n_pairs": len(report), "pairs": report,
              "max_abs_difference_log10_cm_s": max(abs(x["difference_y_minus_log10_cm_s"]) for x in report),
              "max_roundtrip_relative_error": max(x["roundtrip_relative_error"] for x in report),
              "interpretation": "Strong paired evidence for base 10 and 1 cm/s reference; not a complete reconstruction of source aggregation."}
    out = root / "caco_trackA_v1/results/source_audit/label_scale_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
