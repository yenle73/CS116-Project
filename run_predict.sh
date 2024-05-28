#!/bin/bash
echo "Predicting script is running"

model_dir=$1
test_path=$2
output_file=$3

# Chạy quá trình dự đoán
python main.py --mode predict --test_dir "$test_path" --model_output_dir "$model_dir" --output_file "$output_file"

echo "Predicting script execution completed"