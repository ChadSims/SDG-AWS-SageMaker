# SageMaker PyTorch image
FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.6.0-gpu-py312-cu126-ubuntu22.04-sagemaker

ENV PATH="/opt/ml/code:${PATH}"

# Set the working directory to /opt/ml/code
WORKDIR /opt/ml/code

# Install any other dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

# /opt/ml and all subdirectories are utilized by SageMaker, we use the /code subdirectory to store our user code.
RUN mkdir lib
COPY lib ./lib

RUN mkdir scripts
COPY scripts ./scripts

RUN mkdir synthesisers
COPY synthesisers ./synthesisers

COPY run_pipeline.py .

# this environment variable is used by the SageMaker PyTorch container to determine our user code directory.
ENV SAGEMAKER_SUBMIT_DIRECTORY=/opt/ml/code  

# this environment variable is used by the SageMaker PyTorch container to determine our program entry point
# for training and serving.
# For more information: https://github.com/aws/sagemaker-pytorch-container
ENV SAGEMAKER_PROGRAM=run_pipeline.py