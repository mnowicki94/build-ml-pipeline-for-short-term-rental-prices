#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""

import argparse
import logging
import pandas as pd
import wandb


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    artifact_local_path = run.use_artifact(args.input_artifact).file()

    # Read the data
    logger.info("Reading the input artifact")
    df = pd.read_csv(artifact_local_path)

    # Drop outliers
    logger.info("Dropping outliers")
    idx = df["price"].between(args.min_price, args.max_price)
    df = df[idx].copy()

    # Convert last_review to datetime
    logger.info("Converting last_review to datetime")
    df["last_review"] = pd.to_datetime(df["last_review"])

    # Save the cleaned data to a CSV file
    output_file = "cleaned_data.csv"
    logger.info("Saving cleaned data to %s", output_file)
    df.to_csv(output_file, index=False)

    # Log the cleaned data artifact
    logger.info("Logging the cleaned data artifact")
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file(output_file)
    run.log_artifact(artifact)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A very basic data cleaning")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="The name of the input artifact to clean",
        required=True,
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="The name of the output cleaned artifact",
        required=True,
    )

    parser.add_argument(
        "--output_type", type=str, help="The type of the output artifact", required=True
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help="A description for the output artifact",
        required=True,
    )

    parser.add_argument(
        "--min_price",
        type=float,
        help="The minimum price to filter listings",
        required=True,
    )

    parser.add_argument(
        "--max_price",
        type=float,
        help="The maximum price to filter listings",
        required=True,
    )

    args = parser.parse_args()

    go(args)
