#!/bin/bash
echo "Preprocessing Data"
python process_data.py --data_dir $1
echo "Training Models"
python train_model.py --data_dir $1 --results_dir $2
echo "Testing Models"
python test_model.py --data_dir $1 --results_dir $2
