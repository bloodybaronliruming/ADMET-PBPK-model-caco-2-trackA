"""Freeze core baseline search candidates before validation."""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import sklearn
from sklearn.model_selection import ParameterSampler


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "configs/baselines/search_space"
SEARCH_SEED = 20260924
DESCRIPTORS = ["physchem10", "rdkit2d"]
ALL = DESCRIPTORS + ["morgan_r2_2048"]


def freeze(name: str, estimator: str, representations: list[str],
           fixed: dict, candidates: list[dict], search: dict,
           definition_source: str = "endpoint_1_caco_trackA_baseline.md sections 11-14, 30") -> dict:
    assert candidates and len(candidates) == len({json.dumps(x, sort_keys=True) for x in candidates})
    document = {
        "dataset": "caco2_wang",
        "track": "A",
        "model_version": "caco_trackA_v1",
        "algorithm": name,
        "estimator": estimator,
        "applicable_representations": representations,
        "definition_source": definition_source,
        "sklearn_version_at_freeze": sklearn.__version__,
        "fixed_estimator_params": fixed,
        "search": search,
        "candidate_count": len(candidates),
        "candidates": [
            {"candidate_id": f"{name}_{i:02d}", "params": params}
            for i, params in enumerate(candidates, start=1)
        ],
        "validation_reuse": "Same candidates for each applicable representation and all five official splits",
        "validation_random_state_policy": "split_seed for random estimators",
        "training_run": False,
    }
    path = OUTPUT / f"{name.lower()}_candidates.json"
    content = (json.dumps(document, ensure_ascii=False, indent=2) + "\n").encode()
    if path.exists() and path.read_bytes() != content:
        raise FileExistsError(f"Frozen candidate file differs: {path}")
    if not path.exists():
        path.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    print(f"{path.relative_to(ROOT)}: {len(candidates)} candidates, SHA-256 {digest}")
    return {"path": str(path.relative_to(ROOT)), "sha256": digest,
            "candidate_count": len(candidates)}


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    index = {}
    index["ridge"] = freeze(
        "RIDGE", "sklearn.linear_model.Ridge", DESCRIPTORS,
        {"fit_intercept": True},
        [{"alpha": alpha} for alpha in [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]],
        {"method": "full_grid", "grid": {"alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]}},
    )
    svr_grid = {"C": [1, 10, 100], "gamma": ["scale", 0.001, 0.01],
                "epsilon": [0.05, 0.1, 0.2]}
    svr_candidates = [dict(zip(svr_grid, values)) for values in itertools.product(*svr_grid.values())]
    index["svr"] = freeze(
        "SVR", "sklearn.svm.SVR", ALL, {"kernel": "rbf"},
        svr_candidates, {"method": "full_grid", "grid": svr_grid},
    )
    tree_grid = {"n_estimators": [300, 800], "max_depth": [None, 10, 20],
                 "min_samples_leaf": [1, 2, 4], "max_features": ["sqrt", 0.5]}
    sampled = list(ParameterSampler(tree_grid, n_iter=24, random_state=SEARCH_SEED))
    tree_search = {"method": "sklearn.model_selection.ParameterSampler",
                   "grid": tree_grid, "n_iter": 24, "search_seed": SEARCH_SEED,
                   "full_grid_count": 36}
    index["rf"] = freeze(
        "RF", "sklearn.ensemble.RandomForestRegressor", ALL,
        {"criterion": "squared_error", "n_jobs": -1}, sampled, tree_search,
    )
    index["et"] = freeze(
        "ET", "sklearn.ensemble.ExtraTreesRegressor", ALL,
        {"criterion": "squared_error", "n_jobs": -1}, sampled, tree_search,
    )
    xgb_grid = {
        "n_estimators": [300, 600, 1000],
        "learning_rate": [0.02, 0.05, 0.1],
        "max_depth": [3, 5, 7],
        "min_child_weight": [1, 5],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "reg_lambda": [1, 5],
    }
    xgb_sampled = list(ParameterSampler(xgb_grid, n_iter=24, random_state=SEARCH_SEED))
    index["xgb"] = freeze(
        "XGB", "xgboost.XGBRegressor", ALL,
        {"objective": "reg:squarederror", "n_jobs": -1, "device": "cuda", "tree_method": "hist"}, xgb_sampled,
        {"method": "sklearn.model_selection.ParameterSampler",
         "grid": xgb_grid, "n_iter": 24, "search_seed": SEARCH_SEED,
         "full_grid_count": 432},
        "endpoint_1_caco_trackA_baseline.md sections 15, 30",
    )
    manifest = {"scope": "core baseline candidates",
                "search_seed": SEARCH_SEED, "files": index, "model_training": False}
    path = OUTPUT / "candidates_manifest.json"
    content = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode()
    if path.exists() and path.read_bytes() != content:
        old = json.loads(path.read_text())
        if old.get("scope") != "non-XGBoost core baseline candidates" or old.get("search_seed") != SEARCH_SEED or old.get("model_training") is not False or old.get("files") != {key: index[key] for key in ("ridge", "svr", "rf", "et")}:
            raise FileExistsError(f"Frozen manifest differs beyond the planned XGBoost addition: {path}")
        path.write_bytes(content)
    elif not path.exists():
        path.write_bytes(content)
    print(f"Manifest: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
