export HF_HOME='/data/sungjuncho/tmp/' # change to your HF cache directory
MASTER_PORT=28765

for SPLIT in full retain90 retain95 retain99
do
    CUDA_VISIBLE_DEVICES=0,1 torchrun \
    --nproc_per_node=2 \
    --master_port=${MASTER_PORT} \
    finetune.py \
    --config-name=finetune.yaml \
    split=${SPLIT} \
    batch_size=4 \
    gradient_accumulation_steps=4 \
    model_family=llama2-7b \
    lr=1e-5
done

for SPLIT in full retain90 retain95 retain99
do
    CUDA_VISIBLE_DEVICES=0,1 torchrun \
    --nproc_per_node=2 \
    --master_port=${MASTER_PORT} \
    finetune.py \
    --config-name=finetune.yaml \
    split=${SPLIT} \
    batch_size=4 \
    gradient_accumulation_steps=4 \
    model_family=phi \
    lr=2e-5
done