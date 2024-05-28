#!/bin/bash
echo "Training script is running"

train_path=$1
dev_path=$2
model_dir=$3

# Chạy quá trình train mô hình
python main.py --mode train --train_dir "$train_path" --dev_dir "$dev_path" --model_output_dir "$model_dir"
echo "Training script execution completed"
