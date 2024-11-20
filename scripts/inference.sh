#!/bin/bash
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list" 
NUM_GPUS=${#GPULIST[@]}
echo "Active GPUs : ${gpu_list}"

cp -f ../docviz/modeling_llava_next.py /usr/local/lib/python3.10/dist-packages/transformers/models/llava_next/modeling_llava_next.py

for gpu_id in $(seq 0 $((NUM_GPUS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$gpu_id]} python3 ../docviz/inference_doc.py \
    --gpu_id $gpu_id \
    --num_gpus $NUM_GPUS &
done