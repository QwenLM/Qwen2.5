#!/bin/bash

PORT=6069
lsof -i :$PORT | grep LISTEN | awk '{print $2}' | xargs kill -9

nohup python openai_api.py --checkpoint-path Qwen/Qwen1.5-7B-Chat --device 1 --server-port $PORT \
--server-name '0.0.0.0' > nohup.out 2>&1 &

