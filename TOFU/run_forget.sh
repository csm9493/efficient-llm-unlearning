export HF_HOME='/data/sungjuncho/tmp/' # change to your HF cache directory
MASTER_PORT=28765

MODELS=( phi ) # llama2-7b
LR1S=( 2e-05 ) # 1e-05
LR2S=( 0.0002 ) # 0.0001

METHODS=( IHL ) # ( KL idk grad_ascent grad_diff IHL )
BSZS=( 2 ) # ( 1 2 2 2 2 )
GASS=( 16 ) # ( 32 16 16 16 16 )

SPLITS=( forget10 ) # ( forget10 forget05 forget01 )
LORA_TARGETS=( all )

RANKS=( 32 16 8 4 )
DEVICES=( 4 4 4 4 )

for model_idx in "${!MODELS[@]}"
do
    MODEL=${MODELS[$model_idx]}
    LR1=${LR1S[$model_idx]}
    LR2=${LR2S[$model_idx]}
    for method_idx in "${!METHODS[@]}"
    do
        METHOD=${METHODS[$method_idx]}
        BSZ=${BSZS[$method_idx]}
        GAS=${GASS[$method_idx]}
        for lora_target_idx in "${!LORA_TARGETS[@]}"
        do
            LORA_TARGET=${LORA_TARGETS[$lora_target_idx]}
            for split_idx in "${!SPLITS[@]}"
            do
                SPLIT=${SPLITS[$split_idx]}
                for rank_idx in "${!RANKS[@]}"
                do
                    RANK=${RANKS[$rank_idx]}
                    DEVICE=${DEVICES[$rank_idx]}
                    
                    CUDA_VISIBLE_DEVICES=${DEVICE} torchrun \
                    --nproc_per_node=1 \
                    --master_port=${MASTER_PORT} \
                    forget.py \
                    --config-name=forget.yaml \
                    LoRA.targets=${LORA_TARGET} \
                    LoRA.r=${RANK} \
                    LoRA.alpha=$(( ${RANK} * 2 )) \
                    LoRA.dropout=0 \
                    num_epochs=5 \
                    split=${SPLIT} \
                    batch_size=${BSZ} \
                    gradient_accumulation_steps=${GAS} \
                    model_family=${MODEL} \
                    model_path=./llm_weights/ft_epoch5_lr${LR1}_${MODEL}_full_wd0.01/checkpoint-625 \
                    forget_loss=${METHOD} \
                    lr=${LR2} \
                    importance_file=./importances/${MODEL}_${SPLIT}.pt
                done
            done
        done
    done
done