import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


class WOETransformer:
    def __init__(self, numerical_cols=None, categorical_cols=None):
        self.numerical_cols = numerical_cols or []
        self.categorical_cols = categorical_cols or []
        self.woe_dict = {}
        self.iv_dict = {}

    def fit(self, X, y):
        X = X.copy()
        y = pd.Series(y)
        available_num = [col for col in self.numerical_cols if col in X.columns]
        available_cat = [col for col in self.categorical_cols if col in X.columns]

        for col in available_num:
            X[col] = pd.qcut(X[col], q=10, duplicates="drop", labels=False)
            self._calculate_woe(X, y, col)

        for col in available_cat:
            self._calculate_woe(X, y, col)

        return self

    def _calculate_woe(self, X, y, col):
        grouped = X.groupby(col)[y.name].agg(["count", "sum"])
        grouped["non_events"] = grouped["count"] - grouped["sum"]

        total_events = y.sum()
        total_non_events = y.count() - total_events

        grouped["event_rate"] = grouped["sum"] / total_events
        grouped["non_event_rate"] = grouped["non_events"] / total_non_events

        grouped["event_rate"] = grouped["event_rate"].replace(0, 1e-6)
        grouped["non_event_rate"] = grouped["non_event_rate"].replace(0, 1e-6)

        grouped["woe"] = np.log(grouped["event_rate"] / grouped["non_event_rate"])
        grouped["iv"] = (grouped["event_rate"] - grouped["non_event_rate"]) * grouped[
            "woe"
        ]

        self.woe_dict[col] = grouped["woe"].to_dict()
        self.iv_dict[col] = grouped["iv"].sum()

    def transform(self, X):
        X = X.copy()
        available_num = [col for col in self.numerical_cols if col in X.columns]
        available_cat = [col for col in self.categorical_cols if col in X.columns]

        for col in available_num:
            X[col] = pd.qcut(X[col], q=10, duplicates="drop", labels=False)
            X[f"woe_{col}"] = X[col].map(self.woe_dict.get(col, {}))

        for col in available_cat:
            X[f"woe_{col}"] = X[col].map(self.woe_dict.get(col, {}))

        woe_cols = [
            f"woe_{col}"
            for col in (available_num + available_cat)
            if col in self.woe_dict
        ]
        other_cols = [
            col
            for col in X.columns
            if col not in (available_num + available_cat + woe_cols)
        ]
        print(f"WOETransformer output columns: {woe_cols + other_cols}")
        return X[woe_cols + other_cols]


class TemporalFeatureExtractor:
    def __init__(self, datetime_column):
        self.datetime_column = datetime_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if self.datetime_column in X.columns:
            X[self.datetime_column] = pd.to_datetime(X[self.datetime_column])
            X["TransactionHour"] = X[self.datetime_column].dt.hour
            X["TransactionDay"] = X[self.datetime_column].dt.day
            X["TransactionMonth"] = X[self.datetime_column].dt.month
            X["TransactionYear"] = X[self.datetime_column].dt.year
            X = X.drop(columns=[self.datetime_column])
        print(f"TemporalFeatureExtractor output columns: {X.columns.tolist()}")
        return X


class AggregateFeatureExtractor:
    def __init__(self, groupby_column, agg_columns):
        self.groupby_column = groupby_column
        self.agg_columns = agg_columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        available_agg_cols = [col for col in self.agg_columns if col in X.columns]
        if self.groupby_column in X.columns and available_agg_cols:
            agg_dict = {
                col: ["sum", "mean", "count", "std"] for col in available_agg_cols
            }
            agg_df = (
                X.groupby(self.groupby_column, observed=False)
                .agg(agg_dict)
                .reset_index()
            )
            agg_df.columns = [
                f"{col[0]}_{col[1]}" if col[1] else col[0] for col in agg_df.columns
            ]
            X = X.merge(agg_df, on=self.groupby_column)
        print(f"AggregateFeatureExtractor output columns: {X.columns.tolist()}")
        return X


class NumericalTransformer:
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        available_cols = [col for col in self.columns if col in X.columns]
        if "Value" in available_cols:
            X.loc[X["Value"] <= 0, "Value"] = np.nan
        for col in available_cols:
            X[f"log_{col}"] = np.log1p(X[col].abs())
        print(f"NumericalTransformer output columns: {X.columns.tolist()}")
        return X


class LabelEncoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.le = LabelEncoder()

    def fit(self, X, y=None):
        self.le.fit(X)
        return self

    def transform(self, X):
        return self.le.transform(X).reshape(-1, 1)

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self


def create_processing_pipeline():
    numerical_cols = [
        "Amount_sum",
        "Amount_mean",
        "Amount_count",
        "Amount_std",
        "Value_sum",
        "Value_mean",
        "Value_count",
        "Value_std",
        "woe_Amount",
        "woe_Value",
        "woe_TransactionHour",
        "woe_TransactionDay",
        "woe_TransactionMonth",
        "woe_ProductCategory",
        "woe_ChannelId",
    ]
    categorical_cols_le = ["woe_PricingStrategy"]

    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_cols),
            (
                "cat_le",
                LabelEncoderTransformer(),
                categorical_cols_le[0] if categorical_cols_le else None,
            ),
        ]
    )

    pipeline = Pipeline(
        steps=[
            (
                "temporal",
                TemporalFeatureExtractor(datetime_column="TransactionStartTime"),
            ),
            ("numerical", NumericalTransformer(columns=["Amount", "Value"])),
            (
                "aggregate",
                AggregateFeatureExtractor(
                    groupby_column="CustomerId", agg_columns=["Amount", "Value"]
                ),
            ),
            (
                "woe",
                WOETransformer(
                    numerical_cols=[
                        "Amount",
                        "Value",
                        "TransactionHour",
                        "TransactionDay",
                        "TransactionMonth",
                    ],
                    categorical_cols=[
                        "ProductCategory",
                        "ChannelId",
                        "PricingStrategy",
                    ],
                ),
            ),
            ("preprocessor", preprocessor),
        ]
    )

    return pipeline


def process_data(input_path, output_path):
    try:
        data = pd.read_csv(input_path)
        print(f"Input columns: {data.columns.tolist()}")
    except FileNotFoundError:
        print(f"Error: File {input_path} not found.")
        return

    required_columns = [
        "TransactionStartTime",
        "Amount",
        "Value",
        "CustomerId",
        "ProductCategory",
        "ChannelId",
        "PricingStrategy",
        "FraudResult",
    ]
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        print(f"Error: Missing columns in data.csv: {missing_columns}")
        return

    pipeline = create_processing_pipeline()
    try:
        transformed_data = pipeline.fit_transform(data, data["FraudResult"])
    except Exception as e:
        print(f"Error during pipeline transformation: {e}")
        return

    num_features = (
        pipeline.named_steps["preprocessor"]
        .named_transformers_["num"]
        .get_feature_names_out()
        .tolist()
    )
    feature_names = num_features + ["woe_PricingStrategy"]
    print(f"Final output columns: {feature_names}")
    transformed_data_df = pd.DataFrame(transformed_data, columns=feature_names)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    transformed_data_df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")


if __name__ == "__main__":
    input_path = "data/raw/data.csv"
    output_path = "data/processed/processed_data.csv"
    process_data(input_path, output_path)
