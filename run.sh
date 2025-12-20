#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python gen_description.py --shard_id 0 --num_shards 4 &
CUDA_VISIBLE_DEVICES=1 python gen_description.py --shard_id 1 --num_shards 4 &
CUDA_VISIBLE_DEVICES=2 python gen_description.py --shard_id 2 --num_shards 4 &
CUDA_VISIBLE_DEVICES=3 python gen_description.py --shard_id 3 --num_shards 4 &
wait
echo "All shards finished."