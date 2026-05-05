"""
Shared model preprocessing utilities.

Goal: treat champion IDs as categorical via One-Hot Encoding so IDs are not
interpreted as numeric/ordinal values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


POSITIONS = ["top", "jungle", "mid", "adc", "support"]


@dataclass(frozen=True)
class FeatureSpec:
    feature_cols: List[str]
    categorical_cols: List[str]
    numeric_cols: List[str]


def _make_onehot_encoder() -> OneHotEncoder:
    """
    Create a OneHotEncoder that is compatible across sklearn versions.
    """
    try:
        # sklearn >= 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        # sklearn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def infer_feature_spec(
    df: pd.DataFrame,
    *,
    exclude_cols: Sequence[str],
) -> FeatureSpec:
    """
    Infer which columns are model input features, and which of those should be
    treated as categorical champion-ID columns.
    """
    feature_cols = [c for c in df.columns if c not in set(exclude_cols)]

    # Champion-ID like columns:
    # - team1_* / team2_* picks (top/jungle/mid/adc/support)
    # - optional ban columns (if present and not excluded)
    # Exclude derived numeric columns that include these tokens.
    derived_tokens = ("winrate", "matchup", "synergy", "diff")
    position_tokens = tuple(POSITIONS)

    categorical_cols: List[str] = []
    for c in feature_cols:
        c_low = c.lower()
        is_team_pick = ("team1_" in c_low or "team2_" in c_low) and any(p in c_low for p in position_tokens)
        is_ban = "ban" in c_low
        if (is_team_pick or is_ban) and not any(t in c_low for t in derived_tokens):
            categorical_cols.append(c)

    numeric_cols = [c for c in feature_cols if c not in set(categorical_cols)]
    return FeatureSpec(feature_cols=feature_cols, categorical_cols=categorical_cols, numeric_cols=numeric_cols)


def make_preprocessor(spec: FeatureSpec) -> ColumnTransformer:
    """
    Build a transformer that:
    - OneHotEncodes champion ID columns (categorical)
    - Imputes + scales numeric columns
    """
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=-1)),
            ("onehot", _make_onehot_encoder()),
        ]
    )
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("cat", cat_pipe, spec.categorical_cols),
            ("num", num_pipe, spec.numeric_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )



