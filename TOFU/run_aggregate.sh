MODELS=( phi ) # llama2-7b
LR1S=( 2e-05 ) # 1e-05 
LR2S=( 0.0002 ) # 0.0001

METHODS=( IHL_FILA ) # ( KL idk grad_ascent grad_diff IHL )

LORA_TARGETS=( all )
SPLITS=( forget10 ) # ( forget01 forget05 forget10 )
RETAINS=( retain90 ) # ( retain90 retain90 retain90 )
CKPTS=( 562 ) # ( 562 562 562 )

RANKS=( 32 16 8 4 )
EPOCHS=5


for model_idx in "${!MODELS[@]}"
do
    MODEL=${MODELS[$model_idx]}
    LR1=${LR1S[$model_idx]}
    LR2=${LR2S[$model_idx]}

    CKPT_PATH=./llm_weights/ft_epoch5_lr${LR1}_${MODEL}_full_wd0.01/checkpoint-625/eval_results/ds_size300/eval_log_aggregated.json
    RETAIN_PATH=./llm_weights/ft_epoch5_lr${LR1}_${MODEL}_retain90_wd0.01/checkpoint-562/eval_results/ds_size300/eval_log_aggregated.json
    SAVE_FILENAME=./final_results/${MODEL}_base.csv
    
    echo $CKPT_PATH
    echo $RETAIN_PATH
    echo $SAVE_FILENAME
    echo ""
    
    python aggregate_eval_stat.py \
    retain_result=${RETAIN_PATH} \
    ckpt_result=${CKPT_PATH} \
    method_name=base \
    save_file=${SAVE_FILENAME}

    SAVE_FILENAME=./final_results/${MODEL}_retain90.csv
    echo $CKPT_PATH
    echo $RETAIN_PATH
    echo $SAVE_FILENAME
    echo ""
    
    python aggregate_eval_stat.py \
    retain_result=${RETAIN_PATH} \
    ckpt_result=${RETAIN_PATH} \
    method_name=base \
    save_file=${SAVE_FILENAME}

    for method_idx in "${!METHODS[@]}"
    do
        METHOD=${METHODS[$method_idx]}
        for target_idx in "${!LORA_TARGETS[@]}"
        do
            TARGET=${LORA_TARGETS[$target_idx]}
            for rank_idx in "${!RANKS[@]}"
            do
                RANK=${RANKS[$rank_idx]}
                for split_idx in "${!SPLITS[@]}"
                do
                    SPLIT=${SPLITS[$split_idx]}
                    RETAIN=${RETAINS[$split_idx]}
                    CKPT=${CKPTS[$split_idx]}
                    if [ "${SPLIT}" = "forget01" ]
                    then
                        STEPS=( 1 2 3 4 5 )
                    elif [ "${SPLIT}" = "forget05" ]
                    then
                        STEPS=( 6 12 18 24 30 )
                    elif [ "${SPLIT}" = "forget10" ]
                    then
                        STEPS=( 12 24 36 48 60 )
                    else
                        echo "Incorrect split ${SPLIT}"
                    fi
                    
                    for STEP in "${STEPS[@]}"
                    do
                        CKPT_PATH=./llm_weights/ft_epoch5_lr${LR1}_${MODEL}_full_wd0.01/checkpoint-625/${METHOD}_target-${TARGET}_r-${RANK}_${LR2}_${SPLIT}_${EPOCHS}/checkpoint-${STEP}/eval_results/ds_size300/eval_log_aggregated.json
                        RETAIN_PATH=./llm_weights/ft_epoch5_lr${LR1}_${MODEL}_${RETAIN}_wd0.01/checkpoint-${CKPT}/eval_results/ds_size300/eval_log_aggregated.json
                        SAVE_FILENAME=./final_results/${MODEL}_${METHOD}_target-${TARGET}_r-${RANK}_${SPLIT}_step-${STEP}.csv
                        
                        echo $CKPT_PATH
                        echo $RETAIN_PATH
                        echo $SAVE_FILENAME
                        echo ""
                        
                        python aggregate_eval_stat.py \
                        retain_result=${RETAIN_PATH} \
                        ckpt_result=${CKPT_PATH} \
                        method_name=${METHOD} \
                        save_file=${SAVE_FILENAME}
                    done
                done
            done
        done
    done
done
