# Towards Robust and Cost-Efficient Knowledge Unlearning for Large Language Models (ICLR 2025, [Paper Link](https://openreview.net/forum?id=1ExfUpmIW4))
---
![method](assets/method_illustration.png)

This repository is largely derived from the TOFU dataset [repo](https://github.com/locuslab/tofu)

## TOFU Experiments (all steps are run in /TOFU)
---

### 0. Setup Environment
```
conda create -n tofu python=3.10
conda activate tofu
conda install pytorch pytorch-cuda=12.4 -c pytorch -c nvidia
conda install -c "nvidia/label/cuda-12.4.0" cuda-toolkit
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

### 1. Full finetune to obtain base models
```
bash run_finetune.sh
```
Example runs are available in run_finetune.sh

### 2. Unlearn forget set from base models (with LoRA)
```
bash 
```
Example runs are available in run_forget.sh.

### 3. Evaluate unlearned model
```
```
Example runs are available in run_evaluate.sh

### 4. Aggregate evaluation results
```
```
Example runs are available in run_aggregate.sh

## TDEC Experiments
---
In progress... will be uploaded soon!
