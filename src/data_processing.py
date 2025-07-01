import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, TransformerMixin
import os
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class RFMTransformer:
    def __init__(self, snapshot_date=None):
        self.snapshot_date = snapshot_date

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["TransactionStartTime"] = pd.to_datetime(X["TransactionStartTime"])
        snapshot_date = self.snapshot_date or X[
            "TransactionStartTime"
        ].max() + pd.Timedelta(days=1)

        rfm = (
            X.groupby("CustomerId")
            .agg(
                {
                    "TransactionStartTime": lambda x: (snapshot_date - x.max()).days,
                    "TransactionId": "count",
                    "Amount": "sum",
                }
            )
            .reset_index()
        )
        rfm.columns = ["CustomerId", "recency", "frequency", "monetary"]

        scaler = StandardScaler()
        rfm_scaled = scaler.fit_transform(rfm[["recency", "frequency", "monetary"]])

        kmeans = KMeans(n_clusters=3, random_state=42)
        rfm["cluster"] = kmeans.fit_predict(rfm_scaled)

        cluster_summary = rfm.groupby("cluster")[["frequency", "monetary"]].mean()
        high_risk_cluster = cluster_summary.idxmin()["frequency"]
        rfm["is_high_risk"] = (rfm["cluster"] == high_risk_cluster).astype(int)

        X = X.merge(rfm[["CustomerId", "is_high_risk"]], on="CustomerId", how="left")
        X["is_high_risk"] = X["is_high_risk"].fillna(0).astype(int)
        logger.info(f"RFMTransformer output columns: {X.columns.tolist()}")
        return X


class WOETransformer:
    def __init__(self, numerical_cols=None, categorical_cols=None):
        self.numerical_cols = numerical_cols or []
        self.categorical_cols = categorical_cols or []
        self.woe_dict = {}
        self.iv_dict = {}

    def fit(self, X, y):
        logger.info(
            f"Fitting WOETransformer on {len(self.numerical_cols + self.categorical_cols)} columns"
        )
        X = X.copy()
        y = pd.Series(y, name="is_high_risk")
        available_num = [col for col in self.numerical_cols if col in X.columns]
        available_cat = [col for col in self.categorical_cols if col in X.columns]

        for col in available_num:
            try:
                if X[col].nunique() > 1:
                    X[col] = pd.qcut(X[col], q=10, duplicates="drop", labels=False)
                    self._calculate_woe(X, y, col)
                else:
                    logger.warning(
                        f"Assigning default WOE for {col} due to single unique value"
                    )
                    self.woe_dict[col] = {0: 0}
            except Exception as e:
                logger.warning(f"Assigning default WOE for {col} due to error: {e}")
                self.woe_dict[col] = {0: 0}

        for col in available_cat:
            try:
                if X[col].nunique() > 1:
                    self._calculate_woe(X, y, col)
                else:
                    logger.warning(
                        f"Assigning default WOE for {col} due to single unique value"
                    )
                    self.woe_dict[col] = {
                        X[col].iloc[0] if X[col].nunique() > 0 else 0: 0
                    }
            except Exception as e:
                logger.warning(f"Assigning default WOE for {col} due to error: {e}")
                self.woe_dict[col] = {X[col].iloc[0] if X[col].nunique() > 0 else 0: 0}

        return self

    def _calculate_woe(self, X, y, col):
        try:
            grouped = X.groupby(col)[y.name].agg(["count", "sum"])
            grouped["non_events"] = grouped["count"] - grouped["sum"]

            total_events = y.sum()
            total_non_events = y.count() - total_events

            grouped["event_rate"] = grouped["sum"] / total_events
            grouped["non_event_rate"] = grouped["non_events"] / total_non_events

            grouped["event_rate"] = grouped["event_rate"].replace(0, 1e-6)
            grouped["non_event_rate"] = grouped["non_event_rate"].replace(0, 1e-6)

            grouped["woe"] = np.log(grouped["event_rate"] / grouped["non_event_rate"])
            grouped["iv"] = (
                grouped["event_rate"] - grouped["non_event_rate"]
            ) * grouped["woe"]

            self.woe_dict[col] = grouped["woe"].to_dict()
            self.iv_dict[col] = grouped["iv"].sum()
        except Exception as e:
            logger.warning(f"Failed to calculate WOE for {col}: {e}")
            self.woe_dict[col] = {0: 0}

    def transform(self, X):
        logger.info("Transforming data with WOETransformer")
        X = X.copy()
        available_num = [col for col in self.numerical_cols if col in X.columns]
        available_cat = [col for col in self.categorical_cols if col in X.columns]

        for col in available_num:
            try:
                if X[col].nunique() > 1:
                    X[f"woe_{col}"] = (
                        pd.qcut(X[col], q=10, duplicates="drop", labels=False)
                        .map(self.woe_dict.get(col, {}))
                        .fillna(0)
                    )
                else:
                    X[f"woe_{col}"] = 0
                    logger.warning(
                        f"Set default woe_{col} to 0 due to single unique value"
                    )
            except Exception as e:
                X[f"woe_{col}"] = 0
                logger.warning(f"Set default woe_{col} to 0 due to error: {e}")

        for col in available_cat:
            try:
                X[f"woe_{col}"] = X[col].map(self.woe_dict.get(col, {})).fillna(0)
            except Exception as e:
                X[f"woe_{col}"] = 0
                logger.warning(f"Set default woe_{col} to 0 due to error: {e}")

        woe_cols = [
            f"woe_{col}"
            for col in (available_num + available_cat)
            if col in self.woe_dict
        ]
        other_cols = ["CustomerId", "is_high_risk"]
        logger.info(f"WOETransformer output columns: {woe_cols + other_cols}")
        return X[woe_cols + other_cols]


class TemporalFeatureExtractor:
    def __init__(self, datetime_column):
        self.datetime_column = datetime_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        logger.info("Extracting temporal features")
        X = X.copy()
        if self.datetime_column in X.columns:
            X[self.datetime_column] = pd.to_datetime(X[self.datetime_column])
            X["TransactionHour"] = X[self.datetime_column].dt.hour
            X["TransactionDay"] = X[self.datetime_column].dt.day
            X["TransactionMonth"] = X[self.datetime_column].dt.month
            X["TransactionYear"] = X[self.datetime_column].dt.year
            X = X.drop(columns=[self.datetime_column])
        logger.info(f"TemporalFeatureExtractor output columns: {X.columns.tolist()}")
        return X


class AggregateFeatureExtractor:
    def __init__(self, groupby_column, agg_columns):
        self.groupby_column = groupby_column
        self.agg_columns = agg_columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        logger.info("Extracting aggregate features")
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
        logger.info(f"AggregateFeatureExtractor output columns: {X.columns.tolist()}")
        return X


class NumericalTransformer:
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        logger.info("Transforming numerical features")
        X = X.copy()
        available_cols = [col for col in self.columns if col in X.columns]
        if "Value" in available_cols:
            X.loc[X["Value"] <= 0, "Value"] = np.nan
        for col in available_cols:
            X[f"log_{col}"] = np.log1p(X[col].abs())
        logger.info(f"NumericalTransformer output columns: {X.columns.tolist()}")
        return X


def create_processing_pipeline():
    potential_numerical_cols = [
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
        "woe_PricingStrategy",
    ]

    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("rfm", RFMTransformer()),
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
            (
                "preprocessor",
                ColumnTransformer(
                    transformers=[
                        ("num", numerical_transformer, potential_numerical_cols)
                    ]
                ),
            ),
        ]
    )

    return pipeline


def process_data(input_path, output_path):
    try:
        data = pd.read_csv(input_path)
        logger.info(f"Input columns: {data.columns.tolist()}")
        logger.info(
            f"Unique values in input: {data[['Amount', 'Value', 'ProductCategory', 'ChannelId', 'PricingStrategy']].nunique().to_dict()}"
        )
    except FileNotFoundError:
        logger.error(f"File {input_path} not found")
        return

    required_columns = [
        "TransactionStartTime",
        "Amount",
        "Value",
        "CustomerId",
        "ProductCategory",
        "ChannelId",
        "PricingStrategy",
    ]
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        logger.error(f"Missing columns in data.csv: {missing_columns}")
        return

    pipeline = create_processing_pipeline()
    try:
        temp_data = pipeline.named_steps["rfm"].transform(data)
        temp_data = pipeline.named_steps["temporal"].transform(temp_data)
        logger.info(
            f"Unique values after temporal: {temp_data[['Amount', 'Value', 'TransactionHour', 'TransactionDay', 'TransactionMonth', 'ProductCategory', 'ChannelId', 'PricingStrategy']].nunique().to_dict()}"
        )
        woe_data = (
            pipeline.named_steps["woe"]
            .fit(
                pipeline.named_steps["aggregate"].transform(
                    pipeline.named_steps["numerical"].transform(temp_data)
                ),
                temp_data["is_high_risk"],
            )
            .transform(
                pipeline.named_steps["aggregate"].transform(
                    pipeline.named_steps["numerical"].transform(temp_data)
                )
            )
        )
        available_cols = [
            col
            for col in pipeline.named_steps["preprocessor"].transformers[0][2]
            if col in woe_data.columns
        ]
        if not available_cols:
            logger.warning("No WOE columns generated, using raw aggregate features")
            transformed_data_df = woe_data[["CustomerId", "is_high_risk"]]
        else:
            pipeline.named_steps["preprocessor"].transformers = [
                (
                    "num",
                    pipeline.named_steps["preprocessor"].transformers[0][1],
                    available_cols,
                )
            ]
            transformed_data = pipeline.fit_transform(data, temp_data["is_high_risk"])
            num_features = (
                pipeline.named_steps["preprocessor"].get_feature_names_out().tolist()
            )
            transformed_data_df = pd.DataFrame(transformed_data, columns=num_features)
            transformed_data_df["CustomerId"] = data["CustomerId"].values
            transformed_data_df["is_high_risk"] = temp_data["is_high_risk"].values
        logger.info(f"Final output columns: {transformed_data_df.columns.tolist()}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        transformed_data_df.to_csv(output_path, index=False)
        logger.info(f"Processed data saved to {output_path}")
    except Exception as e:
        logger.error(f"Error during pipeline transformation: {e}")


if __name__ == "__main__":
    input_path = "data/raw/data.csv"
    output_path = "data/processed/processed_data.csv"
    process_data(input_path, output_path)
