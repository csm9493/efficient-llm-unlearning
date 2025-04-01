export HF_HOME='/data/sungjuncho/tmp/' # change to your HF cache directory
MASTER_PORT=28765
MODEL=phi # or llama2-7b
LR=2e-05 # or 1e-05

for SPLIT in forget10 forget05 forget01
do
        CUDA_VISIBLE_DEVICES=0 torchrun \
        --nproc_per_node=1 \
        --master_port=${MASTER_PORT} \
        measure_importance.py \
        --config-name=forget.yaml \
        split=${SPLIT} \
        batch_size=4 \
        model_family=${MODEL} \
        model_path=./llm_weights/ft_epoch5_lr${LR}_${MODEL}_full_wd0.01/checkpoint-625
done
