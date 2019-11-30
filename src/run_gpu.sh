#!/bin/sh
# Script to run docker with gpus until "docker-compose run ... gpu ... " potentially works.
docker run --rm --gpus all \
           -v /home/richard/Projects/SHARED/DATASETS:/home/data \
           -v /home/richard/Projects/SHARED/STORAGE:/home/storage \
           -v /etc/timezone:/etc/timezone:ro \
           -v /etc/localtime:/etc/localtime:ro \
           -e PYTHONPATH=/home/src/models:/home/src/util \
           -e UID=1000 \
           -e GID=1000 \
           -ti \
           torchtutorial:gpu bash
