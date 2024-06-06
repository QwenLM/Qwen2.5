#!/usr/bin/env bash
#
# This script will automatically pull docker image from DockerHub, and start a container to run the Qwen-Chat cli-demo.

IMAGE_NAME=qwenllm/qwen:2-cu121
QWEN_CHECKPOINT_PATH=/path/to/Qwen-Instruct
CONTAINER_NAME=qwen2

function usage() {
    echo '
Usage: bash docker/docker_cli_demo.sh [-i IMAGE_NAME] -c [/path/to/Qwen-Instruct] [-n CONTAINER_NAME]
'
}

while [[ "$1" != "" ]]; do
    case $1 in
        -i | --image-name )
            shift
            IMAGE_NAME=$1
            ;;
        -c | --checkpoint )
            shift
            QWEN_CHECKPOINT_PATH=$1
            ;;
        -n | --container-name )
            shift
            CONTAINER_NAME=$1
            ;;
        -h | --help )
            usage
            exit 0
            ;;
        * )
            echo "Unknown argument ${1}"
            exit 1
            ;;
    esac
    shift
done

if [ ! -e ${QWEN_CHECKPOINT_PATH}/config.json ]; then
    echo "Checkpoint config.json file not found in ${QWEN_CHECKPOINT_PATH}, exit."
    exit 1
fi

sudo docker pull ${IMAGE_NAME} || {
    echo "Pulling image ${IMAGE_NAME} failed, exit."
    exit 1
}

sudo docker run --gpus all --rm --name ${CONTAINER_NAME} \
    --mount type=bind,source=${QWEN_CHECKPOINT_PATH},target=/data/shared/Qwen/Qwen-Instruct \
    -it ${IMAGE_NAME} \
    python cli_demo.py -c /data/shared/Qwen/Qwen-Instruct/