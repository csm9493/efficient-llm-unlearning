MASTER_PORT=28765

MODELS=( phi ) # llama2-7b
LR1S=( 2e-05 ) # 1e-05
LR2S=( 0.0002 ) # 0.0001

METHODS=( IHL_FILA ) # ( KL idk grad_ascent grad_diff IHL )
BSZS=( 4 ) # ( 4 4 4 4 4 )

EPOCHS=5
LORA_TARGETS=( all )
SPLITS=( forget10 ) # ( forget10 forget05 forget01 )
RETAINS=( retain90 ) # ( retain90 retain90 retain90 )

RANKS=( 32 16 8 4 )
DEVICE=0

for model_idx in "${!MODELS[@]}"
do
    MODEL=${MODELS[$model_idx]}
    LR1=${LR1S[$model_idx]}
    LR2=${LR2S[$model_idx]}
    BSZ=${BSZS[$model_idx]}

    # run base evaluation
    CUDA_VISIBLE_DEVICES=$DEVICE torchrun \
    --nproc_per_node=1 \
    --master_port=${MASTER_PORT} \
    evaluate_util.py \
    --config-name=eval_everything.yaml \
    batch_size=${BSZ} \
    model_family=${MODEL} \
    split=forget10_perturbed \
    model_path=./llm_weights/ft_epoch5_lr${LR1}_${MODEL}_full_wd0.01/checkpoint-625

    # run retain-only evaluation
    CUDA_VISIBLE_DEVICES=$DEVICE torchrun \
    --nproc_per_node=1 \
    --master_port=${MASTER_PORT} \
    evaluate_util.py \
    --config-name=eval_everything.yaml \
    batch_size=${BSZ} \
    model_family=${MODEL} \
    split=forget10_perturbed \
    model_path=./llm_weights/ft_epoch5_lr${LR1}_${MODEL}_retain90_wd0.01/checkpoint-562

    for method_idx in "${!METHODS[@]}"
    do
        METHOD=${METHODS[$method_idx]}
        for target_idx in "${!LORA_TARGETS[@]}"
        do
            LORA_TARGET=${LORA_TARGETS[$target_idx]}
            for split_idx in "${!SPLITS[@]}"
            do
                SPLIT=${SPLITS[$split_idx]}
    
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
    
                for step_idx in "${!STEPS[@]}"
                
                do
                    STEP=${STEPS[$step_idx]}
                    for rank_idx in "${!RANKS[@]}"
                    do
                        RANK=${RANKS[$rank_idx]}
                        CUDA_VISIBLE_DEVICES=$DEVICE torchrun \
                        --nproc_per_node=1 \
                        --master_port=$(( ${MASTER_PORT} + $rank_idx )) \
                        evaluate_util.py \
                        --config-name=eval_everything.yaml \
                        batch_size=${BSZ} \
                        model_family=${MODEL} \
                        split=${SPLIT}_perturbed \
                        model_path=./llm_weights/ft_epoch5_lr${LR1}_${MODEL}_full_wd0.01/checkpoint-625/${METHOD}_target-${LORA_TARGET}_r-${RANK}_${LR2}_${SPLIT}_${EPOCHS}/checkpoint-${STEP} \
                        base_model_path=./llm_weights/ft_epoch5_lr${LR1}_${MODEL}_full_wd0.01/checkpoint-625/${METHOD}_target-${LORA_TARGET}_r-${RANK}_${LR2}_${SPLIT}_${EPOCHS} \
                        retain_result=./llm_weights/ft_epoch5_lr${LR1}_${MODEL}_retain90_wd0.01/checkpoint-562/eval_results/ds_size300/eval_log_aggregated.json
                    done
                done
            done
        done
    done
done