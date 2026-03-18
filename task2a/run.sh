export GLUE_DIR=$HOME/glue_data
export TASK_NAME=RTE
export GLOO_SOCKET_IFNAME=enp1s0d1

NODE_ID="${HOSTNAME#node-}"
NODE_ID="${NODE_ID%%.*}"
LOCAL_RANK=$NODE_ID

python3 run_glue.py \
  --model_type bert \
  --model_name_or_path bert-base-cased \
  --task_name RTE \
  --do_train \
  --do_eval \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --max_seq_length 128 \
  --per_device_train_batch_size 16 \
  --learning_rate 2e-5 \
  --num_train_epochs 1 \
  --output_dir ./task2a/$TASK_NAME \
  --overwrite_output_dir \
  --world_size 4 \
  --local_rank "$LOCAL_RANK" \
  --master_ip 10.10.1.2 \
  --master_port 12345 \
  --distributed_mode scatter_gather
