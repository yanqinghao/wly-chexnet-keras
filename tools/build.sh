#!/bin/bash

set -e

VERSION=$(bash tools/version.sh)
DOCKERBASES=("suanpan-python-sdk" "suanpan-python-sdk-cuda")
TAG="3.7"
BUILDNAMES=("docker_wly" "docker_wly_gpu")
REQUIREMENTS=("requirements.txt" "requirements_gpu.txt")
NAMESPACE="shuzhi-amd64"
for ((i = 0; i < ${#DOCKERBASES[@]}; i++)); do
    docker build --build-arg NAME_SPACE=${NAMESPACE} --build-arg DOCKER_BASE=${DOCKERBASES[i]} \
        --build-arg PYTHON_VERSION=${TAG} --build-arg REQUIREMENTS_FILE=${REQUIREMENTS[i]} -t \
        registry-vpc.cn-shanghai.aliyuncs.com/${NAMESPACE}/${BUILDNAMES[i]}:${VERSION} \
        -f docker/docker_wly/Dockerfile .
    docker push registry-vpc.cn-shanghai.aliyuncs.com/${NAMESPACE}/${BUILDNAMES[i]}:${VERSION}

    docker tag registry-vpc.cn-shanghai.aliyuncs.com/${NAMESPACE}/${BUILDNAMES[i]}:${VERSION} \
        registry-vpc.cn-shanghai.aliyuncs.com/${NAMESPACE}/${BUILDNAMES[i]}:latest
    docker push registry-vpc.cn-shanghai.aliyuncs.com/${NAMESPACE}/${BUILDNAMES[i]}:latest
done
