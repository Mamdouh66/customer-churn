import numpy as np
import polars as pl

from customer_churn.api.ml.schemas import CustomerData


def create_derived_features(df: pl.DataFrame) -> pl.DataFrame:
    derived_features = [
        (pl.col("total_spend") / pl.col("age")).alias("spend_per_age"),
        (pl.col("usage_frequency") * pl.col("total_spend")).alias("usage_value_index"),
        (pl.col("support_calls") * pl.col("payment_delay")).alias(
            "problem_customer_score"
        ),
        (pl.col("support_calls") > 5).alias("high_support_user"),
        (1 - pl.col("payment_delay") / 30).alias("payment_reliability_score"),
        pl.when(pl.col("total_spend") > 750)
        .then(1)
        .otherwise(0)
        .alias("value_segment_Premium"),
        pl.when((pl.col("total_spend") <= 750) & (pl.col("total_spend") > 250))
        .then(1)
        .otherwise(0)
        .alias("value_segment_Regular"),
    ]

    return df.with_columns(derived_features)


def encode_categorical_features(df: pl.DataFrame) -> pl.DataFrame:
    encoded_features = [
        pl.when(pl.col("gender") == "Male").then(1).otherwise(0).alias("gender_male"),
        pl.when(pl.col("subscription_type") == "Premium")
        .then(1)
        .otherwise(0)
        .alias("subscription_type_premium"),
        pl.when(pl.col("subscription_type") == "Standard")
        .then(1)
        .otherwise(0)
        .alias("subscription_type_standard"),
        pl.when(pl.col("contract_length") == "Monthly")
        .then(1)
        .otherwise(0)
        .alias("contract_length_monthly"),
        pl.when(pl.col("contract_length") == "Quarterly")
        .then(1)
        .otherwise(0)
        .alias("contract_length_quarterly"),
    ]

    return df.with_columns(encoded_features)


def prepare_features(customer: CustomerData) -> np.ndarray:
    df = pl.DataFrame(
        [
            {
                "age": customer.age,
                "gender": customer.gender,
                "tenure": customer.tenure,
                "usage_frequency": customer.usage_frequency,
                "support_calls": customer.support_calls,
                "payment_delay": customer.payment_delay,
                "subscription_type": customer.subscription_type,
                "contract_length": customer.contract_length,
                "total_spend": customer.total_spend,
                "last_interaction": customer.last_interaction,
            }
        ]
    )

    df = create_derived_features(df)
    df = encode_categorical_features(df)

    feature_columns = [
        "gender_male",
        "subscription_type_premium",
        "subscription_type_standard",
        "contract_length_monthly",
        "contract_length_quarterly",
        "spend_per_age",
        "usage_value_index",
        "problem_customer_score",
        "high_support_user",
        "payment_reliability_score",
        "value_segment_Premium",
        "value_segment_Regular",
    ]

    features = df.select(feature_columns).to_numpy()

    return features
