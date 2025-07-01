import pytest
import pandas as pd
from src.data_processing import TemporalFeatureExtractor, AggregateFeatureExtractor


def test_temporal_feature_extractor():
    df = pd.DataFrame(
        {
            "TransactionStartTime": ["2023-01-01 10:00:00", "2023-01-02 15:30:00"],
            "CustomerId": [1, 2],
        }
    )
    transformer = TemporalFeatureExtractor(datetime_column="TransactionStartTime")
    result = transformer.transform(df)
    expected_columns = [
        "CustomerId",
        "TransactionHour",
        "TransactionDay",
        "TransactionMonth",
        "TransactionYear",
    ]
    assert all(col in result.columns for col in expected_columns)
    assert result["TransactionHour"].iloc[0] == 10


def test_aggregate_feature_extractor():
    df = pd.DataFrame({"CustomerId": [1, 1, 2], "Amount": [100, 200, 150]})
    transformer = AggregateFeatureExtractor(
        groupby_column="CustomerId", agg_columns=["Amount"]
    )
    result = transformer.transform(df)
    expected_columns = [
        "CustomerId",
        "Amount_sum",
        "Amount_mean",
        "Amount_count",
        "Amount_std",
    ]
    assert all(col in result.columns for col in expected_columns)
    assert result.loc[result["CustomerId"] == 1, "Amount_sum"].iloc[0] == 300
