#!/bin/bash

export EVAL_FILENAME=../alfred.test.txt
export OUTPUT_NOTE=test-rand


########################################################################################################
# Rand
########################################################################################################

# export RAND=1
# export MODEL_NAME=/data/transformer-vis/output-subsample1-epoch10-rand${RAND}
# python run_evaluation1.py \
#     --model_type=gpt2 \
#     --model_name_or_path=${MODEL_NAME} \
#     --batch_size=32 \
#     --eval_filename=${EVAL_FILENAME} \
#     --output_filename=${OUTPUT_NOTE} \
#     --length=200 \
# 
# export MODEL_NAME=/data/transformer-vis/output-subsample1-epoch20-rand${RAND}
# python run_evaluation1.py \
#     --model_type=gpt2 \
#     --model_name_or_path=${MODEL_NAME} \
#     --batch_size=32 \
#     --eval_filename=${EVAL_FILENAME} \
#     --output_filename=${OUTPUT_NOTE} \
#     --length=200 \
# 
# export MODEL_NAME=/data/transformer-vis/output-subsample1-epoch30-rand${RAND}
# python run_evaluation1.py \
#     --model_type=gpt2 \
#     --model_name_or_path=${MODEL_NAME} \
#     --batch_size=32 \
#     --eval_filename=${EVAL_FILENAME} \
#     --output_filename=${OUTPUT_NOTE} \
#     --length=200 \
#         
# export MODEL_NAME=/data/transformer-vis/output-subsample1-epoch40-rand${RAND}
# python run_evaluation1.py \
#     --model_type=gpt2 \
#     --model_name_or_path=${MODEL_NAME} \
#     --batch_size=32 \
#     --eval_filename=${EVAL_FILENAME} \
#     --output_filename=${OUTPUT_NOTE} \
#     --length=200 \
    

########################################################################################################
# Rand
########################################################################################################

export RAND=1
export MODEL_NAME=/data/transformer-vis/output-subsample01-epoch10-rand${RAND}
python run_evaluation1.py \
    --model_type=gpt2 \
    --model_name_or_path=${MODEL_NAME} \
    --batch_size=32 \
    --eval_filename=${EVAL_FILENAME} \
    --output_filename=${OUTPUT_NOTE} \
    --length=200 \

export MODEL_NAME=/data/transformer-vis/output-subsample01-epoch20-rand${RAND}
python run_evaluation1.py \
    --model_type=gpt2 \
    --model_name_or_path=${MODEL_NAME} \
    --batch_size=32 \
    --eval_filename=${EVAL_FILENAME} \
    --output_filename=${OUTPUT_NOTE} \
    --length=200 \

export MODEL_NAME=/data/transformer-vis/output-subsample01-epoch30-rand${RAND}
python run_evaluation1.py \
    --model_type=gpt2 \
    --model_name_or_path=${MODEL_NAME} \
    --batch_size=32 \
    --eval_filename=${EVAL_FILENAME} \
    --output_filename=${OUTPUT_NOTE} \
    --length=200 \
        
export MODEL_NAME=/data/transformer-vis/output-subsample01-epoch40-rand${RAND}
python run_evaluation1.py \
    --model_type=gpt2 \
    --model_name_or_path=${MODEL_NAME} \
    --batch_size=32 \
    --eval_filename=${EVAL_FILENAME} \
    --output_filename=${OUTPUT_NOTE} \
    --length=200 \
    
########################################################################################################
# Rand
########################################################################################################

export RAND=2
export MODEL_NAME=/data/transformer-vis/output-subsample01-epoch10-rand${RAND}
python run_evaluation1.py \
    --model_type=gpt2 \
    --model_name_or_path=${MODEL_NAME} \
    --batch_size=32 \
    --eval_filename=${EVAL_FILENAME} \
    --output_filename=${OUTPUT_NOTE} \
    --length=200 \

export MODEL_NAME=/data/transformer-vis/output-subsample01-epoch20-rand${RAND}
python run_evaluation1.py \
    --model_type=gpt2 \
    --model_name_or_path=${MODEL_NAME} \
    --batch_size=32 \
    --eval_filename=${EVAL_FILENAME} \
    --output_filename=${OUTPUT_NOTE} \
    --length=200 \

export MODEL_NAME=/data/transformer-vis/output-subsample01-epoch30-rand${RAND}
python run_evaluation1.py \
    --model_type=gpt2 \
    --model_name_or_path=${MODEL_NAME} \
    --batch_size=32 \
    --eval_filename=${EVAL_FILENAME} \
    --output_filename=${OUTPUT_NOTE} \
    --length=200 \
        
export MODEL_NAME=/data/transformer-vis/output-subsample01-epoch40-rand${RAND}
python run_evaluation1.py \
    --model_type=gpt2 \
    --model_name_or_path=${MODEL_NAME} \
    --batch_size=32 \
    --eval_filename=${EVAL_FILENAME} \
    --output_filename=${OUTPUT_NOTE} \
    --length=200 \
    
