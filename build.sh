#!/usr/bin/env bash

# This script shows how to build the Docker image and optionally tag it.
# It does not push the image to ECR, allowing for local inspection.

# The first argument to this script is the image name. This will be used as the
# base image name on the local machine and combined with the account and region
# to form the repository name for ECR.
# The second argument (optional) is the tag for the image. If not provided, "latest"
# will be used.

# Run this script with the image name as the first argument and optionally the
# tag as the second argument. For example:
# ./build.sh my-image
# ./build.sh my-image v1.0

image=$1
tag=$2

if [ "$image" == "" ]
then
    echo "Usage: $0 <image-name> [tag]"
    exit 1
fi

if [ "$tag" == "" ]
then
    tag="latest"
    echo "Using default tag: ${tag}"
else
    echo "Using provided tag: ${tag}"
fi

# Get the account number associated with the current IAM credentials
account=$(aws sts get-caller-identity --query Account --output text)

if [ $? -ne 0 ]
then
    echo "Error getting AWS account ID."
    exit 255
fi

# Define the base region - where image will be pulled from
# Define the target region - where image will be pushed to (used for fullname)
base_region="us-east-1"
target_region="eu-north-1"

fullname="${account}.dkr.ecr.${target_region}.amazonaws.com/${image}:${tag}"

echo "Building image with name: ${image} and tag: ${tag}"
echo "Full ECR name will be: ${fullname} (not pushed)"

# Get the login command from ECR in order to pull down the SageMaker PyTorch image
aws ecr get-login-password --region $base_region | docker login --username AWS --password-stdin 763104351884.dkr.ecr.$base_region.amazonaws.com

# Build the docker image locally with the image name and tag.
docker build -t ${image}:${tag} . --build-arg REGION=${target_region}

# Tag the built image with the full ECR repository name and tag.
docker tag ${image}:${tag} ${fullname}

echo "Docker image built and tagged locally. You can inspect it with:"
echo "docker images | grep '${image}'"
echo "docker inspect ${fullname}"

echo "Image build and tagging complete. The image was NOT pushed to ECR."