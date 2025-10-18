#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

cd "$(dirname $0)"
echo "当前工作目录: $(pwd)"

WRAPPER_CONFIG="tmp_wrapper_${RANDOM}.py"
echo "find_unused_parameters=True" > $WRAPPER_CONFIG
echo "_base_ = ['$CONFIG']" >> $WRAPPER_CONFIG
echo "使用临时包装配置文件: $WRAPPER_CONFIG"
cat $WRAPPER_CONFIG

# 设置Python路径
export PYTHONPATH="$(pwd)":$PYTHONPATH
export PYTHONPATH="$(pwd)/../../":$PYTHONPATH

python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    ./train.py $WRAPPER_CONFIG \
    --launcher pytorch \
    ${@:3}

rm -f $WRAPPER_CONFIG
