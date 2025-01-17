import wandb

artifact = wandb.Artifact(
    args.output_artifact,
    type=args.output_type,
    description=args.output_description,
)
artifact.add_file("clean_sample.csv")
run.log_artifact(artifact)
