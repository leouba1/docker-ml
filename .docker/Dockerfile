# Use the official Python image from the Docker Hub
FROM mcr.microsoft.com/azureml/curated/sklearn-1.1:36

# Set the working directory in the container
WORKDIR /app

# Copy the dependencies file to the working directory
COPY .docker/conda_dependencies.yaml /app/

# Create and activate the Conda environment
RUN conda env create -f /app/conda_dependencies.yaml && \
    conda clean -aqy && \
    rm -rf "$HOME/.cache/pip" && \
    CONDA_ROOT_DIR=$(conda info --root) && \
    rm -rf "$CONDA_ROOT_DIR/pkgs" && \
    find "$CONDA_ROOT_DIR" -type d -name __pycache__ -exec rm -rf {} +

# Copy the rest of the application
COPY src /app/src

# Define environment variable
ENV AZUREML_COMPUTE_WORKER_COUNT=1
ENV PATH /opt/conda/envs/azureml_env/bin:$PATH
