#!/bin/bash

    
####################################################################################
export TRAIN_FILE=/home/peter/github/alfred1/alfred.train.gpt2.txt

export NUM_EPOCHS=10
export OUTPUT_DIR=output-full-epoch${NUM_EPOCHS}
export TEST_FILE=/home/peter/github/alfred1/alfred.dev.gpt2.txt
#--per_gpu_train_batch_size 1 \
#    --learning_rate 1e-4 \
python run_language_modeling.py \
    --output_dir=${OUTPUT_DIR} \
    --model_type=gpt2 \
    --model_name_or_path=gpt2-medium \
    --per_gpu_train_batch_size 1 \
    --num_train_epochs ${NUM_EPOCHS} \
    --save_total_limit 1 \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --eval_data_file=$TEST_FILE


export NUM_EPOCHS=20
export OUTPUT_DIR=output-full-epoch${NUM_EPOCHS}
export TEST_FILE=/home/peter/github/alfred1/alfred.dev.gpt2.txt
#--per_gpu_train_batch_size 1 \
#    --learning_rate 1e-4 \
python run_language_modeling.py \
    --output_dir=${OUTPUT_DIR} \
    --model_type=gpt2 \
    --model_name_or_path=gpt2-medium \
    --per_gpu_train_batch_size 1 \
    --num_train_epochs ${NUM_EPOCHS} \
    --save_total_limit 1 \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --eval_data_file=$TEST_FILE


export NUM_EPOCHS=30
export OUTPUT_DIR=output-full-epoch${NUM_EPOCHS}
export TEST_FILE=/home/peter/github/alfred1/alfred.dev.gpt2.txt
#--per_gpu_train_batch_size 1 \
#    --learning_rate 1e-4 \
python run_language_modeling.py \
    --output_dir=${OUTPUT_DIR} \
    --model_type=gpt2 \
    --model_name_or_path=gpt2-medium \
    --per_gpu_train_batch_size 1 \
    --num_train_epochs ${NUM_EPOCHS} \
    --save_total_limit 1 \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --eval_data_file=$TEST_FILE


export NUM_EPOCHS=40
export OUTPUT_DIR=output-full-epoch${NUM_EPOCHS}
export TEST_FILE=/home/peter/github/alfred1/alfred.dev.gpt2.txt
#--per_gpu_train_batch_size 1 \
#    --learning_rate 1e-4 \
python run_language_modeling.py \
    --output_dir=${OUTPUT_DIR} \
    --model_type=gpt2 \
    --model_name_or_path=gpt2-medium \
    --per_gpu_train_batch_size 1 \
    --num_train_epochs ${NUM_EPOCHS} \
    --save_total_limit 1 \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --eval_data_file=$TEST_FILE


