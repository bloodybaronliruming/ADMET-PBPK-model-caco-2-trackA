"""Record a source-level read boundary audit without opening test data."""

from __future__ import annotations

import ast
import json
from pathlib import Path

from validate_baselines import ROOT, sha256


def read_csv_columns(path: Path) -> list[list[str] | None]:
    tree = ast.parse(path.read_text())
    columns = []
    calls = sorted((node for node in ast.walk(tree)
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "read_csv"), key=lambda node: node.lineno)
    for node in calls:
        choices = [item.value for item in node.keywords if item.arg == "usecols"]
        if not choices:
            if path.name == "evaluate_test.py" and isinstance(node.args[0], ast.Name) and node.args[0].id == "blind_path":
                columns.append(None)
                continue
            raise ValueError(f"{path.name}: read_csv without one explicit usecols")
        if len(choices) != 1:
            raise ValueError(f"{path.name}: multiple usecols arguments")
        names = ast.literal_eval(choices[0])
        if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
            raise ValueError(f"{path.name}: usecols is not a fixed list")
        columns.append(names)
    return columns


def main() -> None:
    scripts = ROOT / "caco_trackA_v1/scripts"
    prediction = scripts / "predict_test.py"
    evaluation = scripts / "evaluate_test.py"
    predict_cols = read_csv_columns(prediction)
    evaluate_cols = read_csv_columns(evaluation)
    if predict_cols != [["Drug_ID"], ["row_id", "source_row"]]:
        raise ValueError(f"Predictor read columns changed: {predict_cols}")
    if evaluate_cols != [None, ["Drug_ID", "Y"]]:
        raise ValueError(f"Evaluator read columns changed: {evaluate_cols}")
    prediction_source = prediction.read_text()
    evaluation_source = evaluation.read_text()
    if ("validation_gate.json" not in prediction_source or "validation_gate.json" not in evaluation_source
            or "frozen_config.json" not in prediction_source or "frozen_config.json" not in evaluation_source
            or "Y_true" not in evaluation_source):
        raise ValueError("Gate or evaluation boundary check missing")
    report = {
        "scope": "source-level audit only; no final test workflow executed",
        "predict_test_sha256": sha256(prediction),
        "evaluate_test_sha256": sha256(evaluation),
        "predict_raw_test_read_columns": ["Drug_ID"],
        "predict_test_feature_source": "frozen feature manifest test matrix",
        "predict_test_y_read": False,
        "evaluate_raw_test_read_columns": ["Drug_ID", "Y"],
        "validation_gate_required_before_prediction_and_evaluation": True,
        "test_file_opened_during_audit": False,
        "blind_or_evaluated_predictions_generated": False,
    }
    path = ROOT / "progress/runs/caco_trackA_prediction_boundary_source_audit.json"
    content = (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode()
    if path.exists() and path.read_bytes() != content:
        raise FileExistsError(f"Existing audit differs: {path}")
    if not path.exists():
        path.write_bytes(content)
    print("Prediction/evaluation source boundary audit passed; fixed test not opened")


if __name__ == "__main__":
    main()
