"""
Unified classification heldout analysis (script).

This script evaluates model predictions on heldout KiNDEL datasets for a
specified `target` and `mode` by computing the ROC AUC of predicted scores
against binary hit labels derived from kd percentiles.

It optionally filters to in-library compounds (those with a defined
`molecule_hash`) and defines binary hit annotations using a percentile
threshold over the kd distribution.
"""

import os
import json
import yaml
import argparse
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def get_testing_data(
    heldout_dir: str,
    target: str,
    mode: str,
    in_library: bool = False,
    hit_percentile: float = 10.0,
) -> Dict[str, pd.DataFrame]:
    """Load heldout KiNDEL data for a target/mode and derive hit labels.

    Parameters
    ----------
    heldout_dir: str
        Directory containing the heldout CSV files, expected to include
        `{target}_ondna.csv` and `{target}_offdna.csv`.
    target: str
        Target identifier (e.g., "ddr1", "mapk14").
    mode: str
        Data mode to load, either "ondna" or "offdna".
    in_library: bool
        If True, restrict to rows where `molecule_hash` is not null (i.e.,
        compounds present in the library only).
    hit_percentile: float
        Percentile over kd values used to define binary hits; compounds with
        kd strictly below this threshold are labeled as hits. Lower kd is
        considered better.

    Returns
    -------
    Dict[str, pd.DataFrame]
        A dictionary keyed by the provided `mode` ("ondna" or "offdna") mapping
        to a DataFrame with columns `smiles`, `y` (kd), and `hits` (0/1).
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
    inference_path: str,
    target: str,
    mode: str,
    in_library: bool,
    hit_percentile: float,
) -> float:
    """Compute ROC AUC between binary hits and predicted scores.

    Parameters
    ----------
    heldout_dir: str
        Directory containing heldout KiNDEL files for the requested target.
    inference_path: str
        CSV file with model predictions for the target; must include columns
        ``X`` (SMILES) and ``y2_preds`` (predicted scores). ``X`` will be
        renamed to ``smiles`` for merging.
    target: str
        Target identifier corresponding to the heldout files.
    mode: str
        Which heldout set to use: "ondna" or "offdna".
    in_library: bool
        Whether to restrict to in-library compounds when loading heldout data.
    hit_percentile: float
        Percentile used when deriving ``hits`` in the heldout set.

    Returns
    -------
    float
        ROC AUC computed over the merged compounds.
    """

    df = pd.read_csv(inference_path)
    df = df.rename(columns={'X': 'smiles'})

    kindel_heldout = get_testing_data(
        heldout_dir=heldout_dir,
        target=target,
        mode=mode,
        in_library=in_library,
        hit_percentile=hit_percentile,
    )
    if mode == "ondna":
        df_kindel_heldout = kindel_heldout["ondna"]
    else:
        df_kindel_heldout = kindel_heldout["offdna"]
    merged_df = pd.merge(df_kindel_heldout, df, on='smiles', how='inner')
    roc_auc_score_val = roc_auc_score(merged_df['hits'], merged_df['y2_preds'])
    return roc_auc_score_val


def main(args: argparse.Namespace) -> None:
    """Entry point that loads config, evaluates, and writes results.

    The configuration YAML pointed to by ``--heldout_config`` must define:
    - ``heldout_dir``: directory with KiNDEL heldout CSVs
    - ``inference_path``: path to predictions CSV (columns ``X``, ``y2_preds``)
    - ``target``: target identifier
    - ``mode``: either "ondna" or "offdna"
    - ``in_library``: boolean flag
    - ``hit_percentile``: float percentile for hit calling
    - ``output``: path to write the evaluation output
    """

    with open(args.heldout_config, 'r') as f:
        heldout_config = yaml.safe_load(f)

    auc_value = evaluate_auc(
        heldout_dir=heldout_config['heldout_dir'],
        inference_path=heldout_config['inference_path'],
        target=heldout_config['target'],
        mode=heldout_config['mode'],
        in_library=heldout_config['in_library'],
        hit_percentile=heldout_config['hit_percentile'],
    )

    os.makedirs(os.path.dirname(os.path.abspath(heldout_config['output']))
                or ".",
                exist_ok=True)
    result = {
        'mode': heldout_config['mode'],
        'target': heldout_config['target'],
        'in_library': heldout_config['in_library'],
        'hit_percentile': heldout_config['hit_percentile'],
        'roc_auc': auc_value
    }
    with open(heldout_config['output'], 'w') as f:
        json.dump(result, f)


if __name__ == "__main__":
    """Entry point that loads config, evaluates, and writes results.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--heldout_config", type=str, required=True)
    args = parser.parse_args()
    main(args)
