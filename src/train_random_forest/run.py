#!/usr/bin/env python
import numpy as np

np.float_ = np.float64
"""
This script trains a Random Forest
"""
import argparse
import logging
import os
import shutil
import matplotlib.pyplot as plt

import mlflow
import json

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, FunctionTransformer

import wandb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline


def delta_date_feature(dates):
    """
    Given a 2d array containing dates (in any format recognized by pd.to_datetime), it returns the delta in days
    between each date and the most recent date in its column
    """
    date_sanitized = pd.DataFrame(dates).apply(pd.to_datetime)
    return date_sanitized.apply(lambda d: (d.max() - d).dt.days, axis=0).to_numpy()


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    run = wandb.init(job_type="train_random_forest")
    run.config.update(args)

    # Get the Random Forest configuration and update W&B
    with open(args.rf_config) as fp:
        rf_config = json.load(fp)
    run.config.update(rf_config)

    # Fix the random seed for the Random Forest, so we get reproducible results
    rf_config["random_state"] = args.random_seed

    # Use run.use_artifact(...).file() to get the train and validation artifact
    logger.info("Fetching trainval artifact")
    trainval_local_path = run.use_artifact(args.trainval_artifact).file()

    X = pd.read_csv(trainval_local_path)
    y = X.pop("price")  # this removes the column "price" from X and puts it into y

    logger.info(f"Minimum price: {y.min()}, Maximum price: {y.max()}")

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=args.val_size,
        stratify=X[args.stratify_by] if args.stratify_by != "none" else None,
        random_state=args.random_seed,
    )

    logger.info("Preparing sklearn pipeline")

    sk_pipe, processed_features = get_inference_pipeline(
        rf_config, args.max_tfidf_features
    )

    # Then fit it to the X_train, y_train data
    logger.info("Fitting pipeline")
    sk_pipe.fit(X_train, y_train)

    # Compute r2 and MAE
    logger.info("Scoring pipeline")
    r_squared = sk_pipe.score(X_val, y_val)

    y_pred = sk_pipe.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)

    logger.info(f"Score: {r_squared}")
    logger.info(f"MAE: {mae}")

    logger.info("Exporting model")

    # Save model package in the MLFlow sklearn format
    if os.path.exists("random_forest_dir"):
        shutil.rmtree("random_forest_dir")

    mlflow.sklearn.save_model(sk_pipe, "random_forest_dir")

    # Upload the model we just exported to W&B
    artifact = wandb.Artifact(
        args.output_artifact,
        type="model_export",
        description="Random Forest model exported",
        metadata=rf_config,
    )
    artifact.add_dir("random_forest_dir")
    run.log_artifact(artifact)

    # Plot feature importance
    fig_feat_imp = plot_feature_importance(sk_pipe, processed_features)

    # Save and log r_squared and MAE
    run.summary["r2"] = r_squared
    run.summary["mae"] = mae

    # Upload to W&B the feature importance visualization
    run.log(
        {
            "feature_importance": wandb.Image(fig_feat_imp),
        }
    )


def plot_feature_importance(pipe, feat_names):
    # We collect the feature importance for all non-nlp features first
    feat_imp = pipe.named_steps["random_forest"].feature_importances_[
        : len(feat_names) - 1
    ]
    # For the NLP feature we sum across all the TF-IDF dimensions into a global
    # NLP importance
    nlp_importance = sum(
        pipe.named_steps["random_forest"].feature_importances_[len(feat_names) - 1 :]
    )
    feat_imp = np.append(feat_imp, nlp_importance)
    fig_feat_imp, sub_feat_imp = plt.subplots(figsize=(10, 10))
    sub_feat_imp.bar(range(feat_imp.shape[0]), feat_imp, color="r", align="center")
    _ = sub_feat_imp.set_xticks(range(feat_imp.shape[0]))
    _ = sub_feat_imp.set_xticklabels(np.array(feat_names), rotation=90)
    fig_feat_imp.tight_layout()
    return fig_feat_imp


def get_inference_pipeline(rf_config, max_tfidf_features):
    ordinal_categorical = ["room_type"]
    non_ordinal_categorical = ["neighbourhood_group"]
    ordinal_categorical_preproc = OrdinalEncoder()

    non_ordinal_categorical_preproc = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder()),
        ]
    )

    zero_imputed = [
        "minimum_nights",
        "number_of_reviews",
        "reviews_per_month",
        "calculated_host_listings_count",
        "availability_365",
        "longitude",
        "latitude",
    ]
    zero_imputer = SimpleImputer(strategy="constant", fill_value=0)

    date_imputer = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="2010-01-01")),
            (
                "delta",
                FunctionTransformer(
                    delta_date_feature, check_inverse=False, validate=False
                ),
            ),
        ]
    )

    reshape_to_1d = FunctionTransformer(np.reshape, kw_args={"newshape": -1})
    name_tfidf = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="")),
            ("reshape", reshape_to_1d),
            (
                "tfidf",
                TfidfVectorizer(
                    binary=False, max_features=max_tfidf_features, stop_words="english"
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("ordinal_cat", ordinal_categorical_preproc, ordinal_categorical),
            (
                "non_ordinal_cat",
                non_ordinal_categorical_preproc,
                non_ordinal_categorical,
            ),
            ("impute_zero", zero_imputer, zero_imputed),
            ("transform_date", date_imputer, ["last_review"]),
            ("transform_name", name_tfidf, ["name"]),
        ],
        remainder="drop",
    )

    processed_features = (
        ordinal_categorical
        + non_ordinal_categorical
        + zero_imputed
        + ["last_review", "name"]
    )

    random_forest = RandomForestRegressor(**rf_config)

    sk_pipe = Pipeline(
        [("preprocessor", preprocessor), ("random_forest", random_forest)]
    )

    return sk_pipe, processed_features


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Basic cleaning of dataset")

    parser.add_argument(
        "--trainval_artifact",
        type=str,
        help="Artifact containing the training dataset. It will be split into train and validation",
    )

    parser.add_argument(
        "--val_size",
        type=float,
        help="Size of the validation split. Fraction of the dataset, or number of items",
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        help="Seed for random number generator",
        default=42,
        required=False,
    )

    parser.add_argument(
        "--stratify_by",
        type=str,
        help="Column to use for stratification",
        default="none",
        required=False,
    )

    parser.add_argument(
        "--rf_config",
        help="Random forest configuration. A JSON dict that will be passed to the "
        "scikit-learn constructor for RandomForestRegressor.",
        default="{}",
    )

    parser.add_argument(
        "--max_tfidf_features",
        help="Maximum number of words to consider for the TFIDF",
        default=10,
        type=int,
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Name for the output serialized model",
        required=True,
    )

    args = parser.parse_args()

    go(args)
