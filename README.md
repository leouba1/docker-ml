# ML Project with Docker and Azure ML
This project demonstrates how to set up, build, and deploy a machine learning model using Docker and Azure Machine Learning (Azure ML). It provides a scalable and maintainable project structure that enables you to run your machine learning workloads both locally and on Azure ML seamlessly.

# Project structure
```markdown
my_ml_project/
├── .azureml/
│   └── config.json
├── .docker/
│   ├── conda_dependencies.yaml
│   └── Dockerfile
├── assets/
│   ├── local_data/
│   │   ├── test.csv
│   │   └── train.csv
│   └── local_outputs/
│       └── model.pkl
├── src/
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── azureml_utils.py
│   │   ├── docker_utils.py
│   │   └── data_utils.py
│   └── training/
│       ├── __init__.py
│       └── train.py
├── run_training.py
├── README.md
├── config.yaml
└── .gitignore
```

## Explanation of the Structure
* .azureml/: Contains Azure ML configuration files.
* .docker/: Contains Docker-related files.
* assets/: Stores data and model outputs, you will need to create these directories to run the training locally.
* src/: Source code for data management, Docker, Azure ML utilities, and training.
* run_training.py: Unified script to build, tag, push Docker images, and run training locally or on Azure ML.

# Build and Push Docker Image
## Build the Docker Image
From the project root directory, run:
```bash
$ docker build -f .docker/Dockerfile -t <your-image-name> .
```

## Tag the Docker Image
Replace <your_acr_name> with your Azure Container Registry name:
```bash
$ docker tag <your-image-name> <your_acr_name>.azurecr.io/<your-image-name>:latest
```

## Push the Docker Image
Log in to Azure Container Registry and push the image:
```bash
$ az acr login --name <your_acr_name>
$ docker push <your_acr_name>.azurecr.io/<your-image-name>:latest
```

# Run Training
## Run Locally
To run the training locally, use:
```bash
python run_training.py --data_path assets/local_data --output_path assets/local_outputs
```
The train script will be executed inside the container, which already contains all the dependencies, by reading data from the `assets/local_data` path and will dump the trained model in `assets/local_outputs`

## Run in Azure ML
To run the training on Azure ML, use:
```bash
python run_training.py --azure
```
This command will trigger a job in Azure ML, by using the provided dataset and will store the model in the job's output.