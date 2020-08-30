#!/bin/bash

chmod +x ./config.sh
. ./config.sh

VER='0a'
F_S=16000
T_D=32
T_S=16
N_SUBBANDS=26
MIN_INSTANCES_SLICE=50
THRESHOLD=0.3
N_WORKERS=18 # used about 25 GB of RAM.

python3 main.py --gpu $GPU                                                \
                --ver $VER                                                \
                --f_s $F_S                                                \
                --T_d $T_D                                                \
                --T_s $T_S                                                \
                --n_subbands $N_SUBBANDS                                  \
                --min_instances_slice $MIN_INSTANCES_SLICE                \
                --threshold $THRESHOLD                                    \
                --n_workers $N_WORKERS                                    \
                --set_path $SET_PATH                                      \
                --noisy_speech_path $NOISY_SPEECH_PATH                    \
                --xi_hat_path $XI_HAT_PATH                                \
                --model_path $MODEL_PATH                                  \
                --data_path $DATA_PATH                                    \
                --train $TRAIN                                            \
                --identification $IDENTIFICATION                          \
                --bounds $BOUNDS                                          \
                --marg $MARG                                              \
