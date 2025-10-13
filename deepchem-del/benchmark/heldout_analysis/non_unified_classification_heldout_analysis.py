"""
Non-unified classification heldout analysis (script).
"""

import os
import json
import yaml
import argparse
import numpy as np
import pandas as pd
from typing import Dict
from sklearn.metrics import roc_auc_score


def get_testing_data(
    heldout_dir: str,
    target: str,
    mode: str,
    in_library: bool = False,
    hit_percentile: float = 10.0,
) -> Dict[str, pd.DataFrame]:
    """Load heldout KiNDEL data and derive binary hits.

    Parameters
    ----------
    heldout_dir: str
        Directory containing `{target}_ondna.csv` and `{target}_offdna.csv`.
    target: str
        Target name (e.g., "ddr1", "mapk14").
    mode: str
        Mode name (e.g., "ondna", "offdna").
    in_library: bool
        If True, filter to rows with non-null `molecule_hash` (in-library only).
    hit_percentile: float
        Percentile for kd threshold to call hits (lower kd is better).

    Returns
    -------
    Dict[str, pd.DataFrame]
        Mapping with keys "ondna" and "offdna" containing dataframes with columns
        `smiles`, `y` (kd), and `hits` (0/1).
    """
    mode_path = os.path.join(heldout_dir, f"{target}_{mode}.csv")
    mode_df = pd.read_csv(mode_path, index_col=0).rename({"kd": "y"},
                                                         axis="columns")
    mode_threshold = float(np.percentile(mode_df["y"], hit_percentile))
    mode_df["hits"] = (mode_df["y"] < mode_threshold).astype(int)

    if in_library:
        # Retain only rows that have `molecule_hash` defined
        if "molecule_hash" in mode_df.columns:
            mode_df = mode_df.dropna(subset=["molecule_hash"
                                             ])  # type: ignore[arg-type]

    return {f"{mode}": mode_df}


def evaluate_auc(
    heldout_dir: str,
    target_inference_path: str,
    matrix_inference_path: str,
    target: str,
    mode: str,
    in_library: bool,
    hit_percentile: float,
) -> float:
    """
    Compute ROC AUC for the product of per-compound target and matrix predictions.

    Parameters
    ----------
    heldout_dir: str
        Directory containing heldout CSVs for the specified target.
    target_inference_path: str
        Path to CSV with target model predictions containing columns `smiles`, `y2_preds`.
    matrix_inference_path: str
        Path to CSV with matrix model predictions containing column `y1_preds` aligned to `smiles`.
    target: str
        Target name used to locate heldout evaluation files.
    mode: str
        Either "ondna" or "offdna"; selects which heldout file to evaluate against.
    in_library: bool
        If True, restrict evaluation to compounds present in-library (non-null `molecule_hash`).
    hit_percentile: float
        Percentile threshold applied on kd to define binary hits (lower kd is better).

    Returns
    -------
    float
        ROC AUC of the combined prediction (`hit_t * hit_m`) against binary hits.
    """

    df_t = pd.read_csv(target_inference_path)
    df_m = pd.read_csv(matrix_inference_path)
    df_t["hit_t"] = df_t["y2_preds"]
    df_m["hit_m"] = df_m["y1_preds"]
    df_comb = pd.concat([df_t[["X", "hit_t"]], df_m[["hit_m"]]], axis=1)
    df_comb["net_result"] = df_comb.apply(lambda x: x.hit_t * x.hit_m, axis=1)
    df_comb = df_comb.rename(columns={'X': 'smiles'})

    kindel_headout = get_testing_data(
        heldout_dir=heldout_dir,
        target=target,
        mode=mode,
        in_library=in_library,
        hit_percentile=hit_percentile,
    )
    if mode == "ondna":
        df_kindel_headout = kindel_headout["ondna"]
    else:
        df_kindel_headout = kindel_headout["offdna"]
    merged_df = pd.merge(df_kindel_headout, df_comb, on='smiles', how='inner')
    roc_auc_score_val = roc_auc_score(merged_df['hits'],
                                      merged_df['net_result'])
    return roc_auc_score_val


def main(args: argparse.Namespace) -> None:
    """
    Entry point: load config, evaluate AUC, and persist a one-row CSV result.

    Parameters
    ----------
    args
        Parsed command-line arguments containing `heldout_config` path.
    """

    with open(args.heldout_config, 'r') as f:
        heldout_config = yaml.safe_load(f)

    df_results = evaluate_auc(
        heldout_dir=heldout_config['heldout_dir'],
        target_inference_path=heldout_config['target_inference_path'],
        matrix_inference_path=heldout_config['matrix_inference_path'],
        target=heldout_config['target'],
        mode=heldout_config['mode'],
        in_library=heldout_config['in_library'],
        hit_percentile=heldout_config['hit_percentile'],
    )

    result = {
        'mode': heldout_config['mode'],
        'target': heldout_config['target'],
        'in_library': heldout_config['in_library'],
        'hit_percentile': heldout_config['hit_percentile'],
        'roc_auc': df_results
    }

    os.makedirs(os.path.dirname(os.path.abspath(heldout_config['output']))
                or ".",
                exist_ok=True)
    # save the results in a json file
    with open(heldout_config['output'], 'w') as f:
        json.dump(result, f)

    # with pd.option_context("display.max_rows", None, "display.max_columns", None):
    #     print(df_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--heldout_config", type=str, required=True)
    args = parser.parse_args()
    main(args)
