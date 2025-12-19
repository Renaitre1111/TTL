#!/bin/bash

# Default parameters
DATA_ROOT='datasets/' # '/home/raza.imam/Documents/TPT/datasets'
TEST_SETS='A' # Options: A/V/R/K for out-of-domain classification
MODE='test'
ARCH='./weights/ViT-B-16.pt' # Options: ViT-B/16, ViT-B/32, RN50
BS=64
CTX_INIT='a_photo_of_a' #prompt initialization
LR=5e-3
TTA_STEPS=1 # Number of Adaptation steps for 1 input
PRINT_FRQ=50
GPU=1 # GPU ID
SELECTION_P=1.0 # Options: 0.1=6 augs whereas 1.0=64 augs
LAYER_RANGE=9,11 # Apply LORA on layers 9-11
INIT_METHOD='xavier' # weight initializations
LORA_ENCODER='image' # Options: image, text, prompt (for TPT)
RANK=16 #LoRA weight ranks
DEYO_SELECTION=True # Options: True, False (True for weighted entropy)
DEYO_MARGIN_E0=0.4



# Command to run ttl.py with the above parameters
python3 main.py --data $DATA_ROOT \
               --test_sets $TEST_SETS \
               --dataset_mode $MODE \
               --arch $ARCH \
               --b $BS \
               --ctx_init $CTX_INIT \
               --lr $LR \
               --tta_steps $TTA_STEPS \
               --print_freq $PRINT_FRQ \
               --gpu $GPU \
               --selection_p $SELECTION_P \
               --layer_range $LAYER_RANGE \
               --init_method $INIT_METHOD \
               --lora_encoder $LORA_ENCODER \
               --rank $RANK \
               --deyo_selection $DEYO_SELECTION \
               --deyo_margin_e0 $DEYO_MARGIN_E0 \
               --mc_lora