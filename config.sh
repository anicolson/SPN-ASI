#!/bin/bash

PROJ_DIR='spn_asi'

set -o noglob

case `hostname` in
"stink")  echo "Running on `hostname`."
		DATA_PATH='/home/aaron/mnt/fist/data/'$PROJ_DIR
		SET_PATH='/home/aaron/mnt/aaron_root/mnt/hdd1/set/timit_wav/timit'
		NOISY_SPEECH_PATH='/home/aaron/mnt/aaron/set/spn_asi_20/noisy_speech'
		XI_HAT_PATH='/home/aaron/mnt/aaron/set/spn_asi_20/xi_hat_deep_xi_3e_e150'
		MODEL_PATH='/home/aaron/mnt/fist/model/'$PROJ_DIR
    ;;
"pinky-jnr")  echo "Running on `hostname`."
		DATA_PATH='/home/aaron/mnt/fist/data/'$PROJ_DIR
		SET_PATH='/home/aaron/mnt/aaron_root/mnt/hdd1/set/timit_wav/timit'
		NOISY_SPEECH_PATH='/home/aaron/mnt/aaron/set/spn_asi_20/noisy_speech'
		XI_HAT_PATH='/home/aaron/mnt/aaron/set/spn_asi_20/xi_hat_deep_xi_3e_e150'
		MODEL_PATH='/home/aaron/mnt/fist/model/'$PROJ_DIR
    ;;
*) echo "`hostname` is not a known workstation. Using default paths."
		DATA_PATH='data'
		SET_PATH='timit'
		NOISY_SPEECH_PATH='spn_asi_20/noisy_speech'
		XI_HAT_PATH='spn_asi_20/xi_hat_deep_xi_3e_e150'
		MODEL_PATH='model'
    ;;
esac

get_free_gpu () {
    NUM_GPU=$( nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader | wc -l )
    echo "$NUM_GPU total GPU/s."
    if [ $1 -eq 1  ]
    then
        echo 'Sleeping'
        sleep 1m
    fi
    while true
    do
        for (( gpu=0; gpu<$NUM_GPU; gpu++ ))
        do
            VAR1=$( nvidia-smi -i $gpu --query-gpu=pci.bus_id --format=csv,noheader )
            VAR2=$( nvidia-smi -i $gpu --query-compute-apps=gpu_bus_id --format=csv,noheader | head -n 1)
            if [ "$VAR1" != "$VAR2" ]
            then
                return $gpu
            fi
        done
        echo 'Waiting for free GPU.'
        sleep 1m
    done
}

TRAIN=0
IDENTIFICATION=0
MARG=0
BOUNDS=0

for ARGUMENT in "$@"
do
    KEY=$(echo $ARGUMENT | cut -f1 -d=)
    VALUE=$(echo $ARGUMENT | cut -f2 -d=)
    case "$KEY" in
            GPU)                 GPU=${VALUE} ;;
            TRAIN)               TRAIN=${VALUE} ;;
						IDENTIFICATION)      IDENTIFICATION=${VALUE} ;;
						MARG)                MARG=${VALUE} ;;
						BOUNDS)              BOUNDS=${VALUE} ;;
            *)
    esac
done

WAIT=0
if [ -z $GPU ]
then
    get_free_gpu $WAIT
    GPU=$?
fi
