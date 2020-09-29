#!/bin/bash
#--model_name_or_path=/data/github/transformers/examples/language-modeling/output-epoch1 \
#--model_name_or_path=/home/peter/github/transformers.vis/examples/language-modeling/output-epoch25 \
#--model_name_or_path=/home/peter/transformers-vis/output-epoch1 \
#export MODEL_NAME=/home/peter/transformers-vis/output-epoch1


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
        
