#!/bin/bash

# env variable for multiple GPUs
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
# export NCCL_SOCKET_IFNAME=eth1

export TORCH_HOME='/path/to/torch/'     # save pretrained models

. ~/.bashrc

# distributed setting
N_NODES="1"
NODE_RANK="0"
MASTER_ADDR="localhost"
MASTER_PORT="12345"
GPUS_PER_NODE="1"

WORKSPACE='/path/to/checkpoints/NPVA/data/'
VERSION='name'

DEFAULT_PARAMS="./configs/0426/full.yaml"

/data/miniconda3/envs/npva/bin/python3 -m torch.distributed.launch \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    --nnodes $N_NODES \
    --nproc_per_node $GPUS_PER_NODE \
    --node_rank $NODE_RANK train.py \
    --config_path $DEFAULT_PARAMS \
    --workspace $WORKSPACE --version $VERSION \
    --extra_config '{"training.gpus": "0"}'