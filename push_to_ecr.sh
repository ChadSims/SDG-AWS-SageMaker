#!/usr/bin/env bash

# This script shows how to build the Docker image and push it to ECR to be ready for use
# by SageMaker.

# The argument to this script is the image name. This will be used as the image on the local
# machine and combined with the account and region to form the repository name for ECR.

# Run this script with the image name as the argument. For example:
# ./build_and_push.sh my-image

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

# If the repository doesn't exist in ECR, create it.

aws ecr describe-repositories --repository-names "${image}" --region "${target_region}" > /dev/null 2>&1

if [ $? -ne 0 ]
then
    aws ecr create-repository --repository-name "${image}" --region "${target_region}" > /dev/null
fi

# Get the login command from ECR and execute it directly
aws ecr get-login-password --region $target_region | docker login --username AWS --password-stdin $account.dkr.ecr.$target_region.amazonaws.com


echo "Pushing image ${fullname} to ECR"
docker push ${fullname}