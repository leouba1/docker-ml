import os
import argparse
import subprocess
import yaml
from src.utils.azureml_utils import get_workspace, submit_experiment

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def run_locally(data_path, output_path, image_name):
    command = [
        "docker", "run",
        "-v", f"{os.path.abspath(data_path)}:/data",
        "-v", f"{os.path.abspath(output_path)}:/outputs",
        image_name,
        "python", "src/training/train.py",
        "--data_path", "/data",
        "--output_path", "/outputs"
    ]
    subprocess.run(command, check=True)

def run_in_azure(config):
    ws = get_workspace()
    submit_experiment(
        ws=ws,
        experiment_name=config["experiment_name"],
        source_directory=config["source_directory"],
        script=config["script"],
        arguments=config["arguments"],
        dataset_name=config["dataset_name"],
        acr_name=config["acr_name"],
        image_name=config["image_name"]
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the configuration file")
    parser.add_argument("--data_path", type=str, default="assets/local_data", help="Path to training data")
    parser.add_argument("--output_path", type=str, default="assets/local_outputs", help="Path to save the trained model")
    parser.add_argument("--azure", action="store_true", help="Run in Azure ML")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.azure:
        run_in_azure(config)
    else:
        run_locally(args.data_path, args.output_path, config["image_name"])
