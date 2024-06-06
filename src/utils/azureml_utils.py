from azureml.core import Workspace, Experiment, ScriptRunConfig, Environment, Dataset
from azureml.core.runconfig import DockerConfiguration

def get_workspace():
    return Workspace.from_config()

def get_experiment(ws, experiment_name):
    return Experiment(ws, experiment_name)

def create_docker_environment(acr_name, image_name):
    docker_config = DockerConfiguration(use_docker=True)
    env = Environment(name="local-docker-env")
    env.docker.base_image = f"{acr_name}.azurecr.io/{image_name}:latest"
    return env, docker_config

def get_dataset(ws, dataset_name):
    return Dataset.get_by_name(ws, dataset_name)

def submit_experiment(ws, experiment_name, source_directory, script, arguments, dataset_name, acr_name, image_name):
    experiment = get_experiment(ws, experiment_name)
    env, docker_config = create_docker_environment(acr_name, image_name)
    dataset = get_dataset(ws, dataset_name)

    src = ScriptRunConfig(
        source_directory=source_directory,
        script=script,
        arguments=arguments,
        environment=env,
        docker_runtime_config=docker_config
    )

    src.run_config.data = {'training_data': dataset.as_named_input('training_data')}

    run = experiment.submit(src)
    run.wait_for_completion(show_output=True)
