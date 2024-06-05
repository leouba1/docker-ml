from azureml.core import Workspace, Experiment, ScriptRunConfig, Environment, Dataset
from azureml.core.runconfig import DockerConfiguration

ws = Workspace.from_config()
experiment = Experiment(ws, "cloud-docker-test")

docker_config = DockerConfiguration(use_docker=True)
env = Environment(name="local-docker-env")
env.docker.base_image = "97d0214391784f41b98f211514d8c298.azurecr.io/azureml-local-test:latest"
env.python.user_managed_dependencies = True

# Get the dataset (ensure the dataset is registered in your workspace)
dataset = Dataset.get_by_name(ws, "titanic_train_data")

# Define the configuration for the script run
src = ScriptRunConfig(
    source_directory='src',
    script='training/train.py',
    arguments=['--data_path', '/data', '--output_path', './outputs'],
    environment=env,
    docker_runtime_config=docker_config
)

# Assign the dataset to a named input 'training_data' for the script
src.run_config.data = {'training_data': dataset.as_named_input('training_data')}

# Submit the run
run = experiment.submit(src)
run.wait_for_completion(show_output=True)
