#!/bin/bash

export EVAL_FILENAME=../alfred.test.txt
export OUTPUT_NOTE=test-full

########################################################################################################

export MODEL_NAME=/data/transformer-vis/output-full-epoch10
python run_evaluation1.py \
    --model_type=gpt2 \
    --model_name_or_path=${MODEL_NAME} \
    --batch_size=32 \
    --eval_filename=${EVAL_FILENAME} \
    --output_filename=${OUTPUT_NOTE} \
    --length=200 \

export MODEL_NAME=/data/transformer-vis/output-full-epoch20
python run_evaluation1.py \
    --model_type=gpt2 \
    --model_name_or_path=${MODEL_NAME} \
    --batch_size=32 \
    --eval_filename=${EVAL_FILENAME} \
    --output_filename=${OUTPUT_NOTE} \
    --length=200 \

export MODEL_NAME=/data/transformer-vis/output-full-epoch30
python run_evaluation1.py \
    --model_type=gpt2 \
    --model_name_or_path=${MODEL_NAME} \
    --batch_size=32 \
    --eval_filename=${EVAL_FILENAME} \
    --output_filename=${OUTPUT_NOTE} \
    --length=200 \
        
