"""
Helper functions and custom transformers for the Spaceship Titanic pipeline
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


SPENDING_COLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]

CATEGORICAL_FEATURES = [
    "HomePlanet",
    "CryoSleep",
    "Destination",
    "VIP",
    "Deck",
    "Side",
    "Age_group",
]

NUMERICAL_FEATURES = [
    "Age",
    "RoomService",
    "FoodCourt",
    "ShoppingMall",
    "Spa",
    "VRDeck",
    "Cabin_num",
    "Group_size",
    "Solo",
    "Family_size",
    "TotalSpending",
    "HasSpending",
    "NoSpending",
    "Age_missing",
    "CryoSleep_missing",
    "RoomService_ratio",
    "FoodCourt_ratio",
    "ShoppingMall_ratio",
    "Spa_ratio",
    "VRDeck_ratio",
]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Derives features from raw Spaceship Titanic data"""

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df = X.copy()

        # Cabin features
        df["Deck"] = df["Cabin"].apply(
            lambda x: x.split("/")[0] if pd.notna(x) else "Unknown"
        )
        df["Cabin_num"] = (
            df["Cabin"]
            .apply(lambda x: x.split("/")[1] if pd.notna(x) else np.nan)
            .astype(float)
        )
        df["Side"] = df["Cabin"].apply(
            lambda x: x.split("/")[2] if pd.notna(x) else "Unknown"
        )

        # Group and solo traveller flags
        df["Group"] = df["PassengerId"].apply(lambda x: x.split("_")[0])
        df["Group_size"] = df.groupby("Group")["Group"].transform("count")
        df["Solo"] = (df["Group_size"] == 1).astype(int)

        # Family size from last name
        df["LastName"] = df["Name"].apply(
            lambda x: x.split()[-1] if pd.notna(x) else "Unknown"
        )
        df["Family_size"] = df.groupby("LastName")["LastName"].transform("count")

        # Spending aggregates
        df["TotalSpending"] = df[SPENDING_COLS].sum(axis=1)
        df["HasSpending"] = (df["TotalSpending"] > 0).astype(int)
        df["NoSpending"] = (df["TotalSpending"] == 0).astype(int)

        for col in SPENDING_COLS:
            df[f"{col}_ratio"] = df[col] / (df["TotalSpending"] + 1)

        # Age group buckets
        df["Age_group"] = (
            pd.cut(
                df["Age"],
                bins=[0, 12, 18, 30, 50, 100],
                labels=["Child", "Teen", "Young_Adult", "Adult", "Senior"],
            )
            .astype(str)
            .replace("nan", "Unknown")
        )

        # Missing value indicators
        df["Age_missing"] = df["Age"].isna().astype(int)
        df["CryoSleep_missing"] = df["CryoSleep"].isna().astype(int)

        # CryoSleep and VIP arrive as mixed bool/NaN, OrdinalEncoder hates that
        for col in ["CryoSleep", "VIP"]:
            df[col] = df[col].astype(str)
            df[col] = df[col].replace("nan", np.nan)

        return df


def _categorical_pipeline():
    """Impute missing with Unknown then OrdinalEncode"""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encoder", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
    ])


def _numerical_pipeline():
    """Impute missing with median then scale"""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])


def _preprocessor():
    return ColumnTransformer(
        transformers=[
            ("cat", _categorical_pipeline(), CATEGORICAL_FEATURES),
            ("num", _numerical_pipeline(), NUMERICAL_FEATURES),
        ],
        remainder="drop",
    )


def make_pipeline(classifier=None):
    """
    Returns an end-to-end sklearn Pipeline

    Pass a classifier or get LogisticRegression by default
    """
    if classifier is None:
        classifier = LogisticRegression(max_iter=2000, random_state=42)

    return Pipeline([
        ("feature_engineer", FeatureEngineer()),
        ("preprocessor", _preprocessor()),
        ("classifier", classifier),
    ])
