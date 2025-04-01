from data_module import TextForgetDatasetQA, TextForgetDatasetDPOQA
from dataloader import CustomTrainerForgetting, custom_data_collator_forget
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, set_seed
import deepspeed
import json
import hydra 
import transformers
import os
from functools import reduce
from peft import LoraConfig, get_peft_model, PeftModel, load_peft_weights, set_peft_model_state_dict
from pathlib import Path
from utils import get_model_identifiers_from_yaml
from omegaconf import OmegaConf

import random
import numpy as np
from tqdm import tqdm

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

@hydra.main(version_base=None, config_path="config", config_name="forget")
def main(cfg):

    num_devices = int(os.environ.get('WORLD_SIZE', 1))
    print(f"num_devices: {num_devices}")

    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}

    set_seed(cfg.seed)

    os.environ["WANDB_DISABLED"] = "true"
    model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    model_id = model_cfg["hf_key"]
    if cfg.model_path is None:
        cfg.model_path = model_cfg["ft_model_path"]

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    max_length = 500
    torch_format_dataset = TextForgetDatasetQA(
        cfg.data_path, 
        tokenizer=tokenizer, 
        model_family=cfg.model_family, 
        max_length=max_length, 
        split=cfg.split, 
        loss_type='grad_diff'
    )
    
    batch_size = cfg.batch_size
    gradient_accumulation_steps = cfg.gradient_accumulation_steps
    steps_per_epoch = len(torch_format_dataset)//(batch_size*gradient_accumulation_steps*num_devices)

    max_steps = int(cfg.num_epochs*len(torch_format_dataset))//(batch_size*gradient_accumulation_steps*num_devices)
    print(f"max_steps: {max_steps}")

    training_args = transformers.TrainingArguments(
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=max(1, steps_per_epoch),
        max_steps=max_steps,
        learning_rate=cfg.lr,
        bf16=True,
        bf16_full_eval=True,
        logging_steps=max(1,max_steps//20),
        optim="paged_adamw_32bit",
        save_strategy="steps" if cfg.save_model and (not cfg.eval_only) else "no",
        save_steps=steps_per_epoch,
        save_only_model=True,
        ddp_find_unused_parameters= False,
        deepspeed='config/ds_config.json',
        weight_decay = cfg.weight_decay,
        eval_steps = steps_per_epoch,
        evaluation_strategy = "steps" if cfg.eval_while_train else "no",
        seed=cfg.seed
    )
    
    #first get the base model architectur2e
    #if there is a pytorch*.bin file in the model path, then load that. use regex there can be anythign in between pytorch and .bin
    import re
    path_found = False
    for file in os.listdir(cfg.model_path):
        if re.search("pytorch.*\.bin", file):
            path_found = True
            break
        
        if re.search("model-*\.safetensors", file):
            path_found = True
            break

    oracle_model = None

    if path_found:
        config = AutoConfig.from_pretrained(model_id)

        print("Loading from checkpoint")
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_path, 
            config=config, 
            use_flash_attention_2=model_cfg["flash_attention2"]=="true", 
            torch_dtype=torch.bfloat16, 
            trust_remote_code = True
        )

    else:
        print("Loading after merge and unload")
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            use_flash_attention_2=model_cfg["flash_attention2"]=="true", 
            torch_dtype=torch.bfloat16, 
            device_map=device_map
        )
        #now use the checkpoint to add the LoRA modules
        model = PeftModel.from_pretrained(model, model_id = cfg.model_path)
        #save this as a standard model so that we can again do PEFT style finetuneing from scratch
        model = model.merge_and_unload()
        #save the model for next time
        model.save_pretrained(cfg.model_path)
    
    
    # Hot fix for https://discuss.huggingface.co/t/help-with-llama-2-finetuning-setup/50035
    model.generation_config.do_sample = True
    
    #now we have a HuggingFace model 
    if model_cfg["gradient_checkpointing"] == "true":
        model.gradient_checkpointing_enable()
    
    trainer = CustomTrainerForgetting(
        model=model,
        tokenizer=tokenizer,
        train_dataset=torch_format_dataset,
        eval_dataset=torch_format_dataset,
        compute_metrics=None, # the callback for computing metrics, None in this case since you're doing it in your callback
        # callbacks=[GlobalStepDeletionCallback],
        args=training_args,
        data_collator=custom_data_collator_forget,
        oracle_model = oracle_model,
        forget_loss = 'grad_diff',
        eval_cfg = cfg.eval,
        seed = cfg.seed, # for NPO
        ref_policy = cfg.ref_policy,
        beta = cfg.beta,
        npo_coeff=cfg.npo_coeff,
        grad_diff_coeff=cfg.grad_diff_coeff,
        KL_coeff=cfg.KL_coeff,
    )
    
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    
    import copy
    deepspeed_plugin = trainer.accelerator.state.deepspeed_plugin
    config_kwargs = copy.deepcopy(deepspeed_plugin.deepspeed_config)
    if model is not None:
        if hasattr(model, "config"):
            hidden_size = (
                max(model.config.hidden_sizes)
                if getattr(model.config, "hidden_sizes", None)
                else getattr(model.config, "hidden_size", None)
            )
            if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                config_kwargs.update(
                    {
                        "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                        "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                        "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                    }
                )

    # If ZeRO-3 is used, we shard both the active and reference model.
    # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
    if config_kwargs["zero_optimization"]["stage"] != 3:
        config_kwargs["zero_optimization"]["stage"] = 0
    config_kwargs["optimizer"] = {"type": None}
    ds_model, *_ = deepspeed.initialize(model=model, config=config_kwargs)

    # Force all parameters to require gradients
    ds_model.train()
    for param in ds_model.parameters():
        param.requires_grad = True

    # Find all names of linear weights
    target_modules = find_all_linear_names(ds_model)
    
    ##### GET IMPORTANCES
    importance_f = {}
    importance_r = {}
    for name, param in ds_model.named_parameters():
        for t in target_modules:
            if t in name and 'weight' in name:
                importance_f[name] = 0
                importance_r[name] = 0


    f_cnt = 0
    r_cnt = 0
    for epochs in range(1): # use 1-epoch importance measurement for now
        for step, inputs in tqdm(enumerate(trainer.get_train_dataloader())):
    
            forget_input, retain_input = inputs
            
            input_ids, labels, attention_mask = forget_input
            output = ds_model(
                input_ids.to(model.device),
                labels=labels.to(model.device),
                attention_mask=attention_mask.to(model.device)
            )
            output.loss.backward()
            cnt = torch.sum(labels != -100)
            for n, lp in ds_model.named_parameters():
                if n in importance_f:
                    importance_f[n] += (lp.grad.pow(2) * cnt).detach().cpu()
                lp.grad = None
            f_cnt += cnt
    
            input_ids, labels, attention_mask = retain_input
            output = ds_model(
                input_ids.to(model.device),
                labels=labels.to(model.device),
                attention_mask=attention_mask.to(model.device)
            )
            output.loss.backward()
            cnt = torch.sum(labels != -100)
            for n, lp in ds_model.named_parameters():
                if n in importance_r:
                    importance_r[n] += (lp.grad.pow(2) * cnt).detach().cpu()
                lp.grad = None
            r_cnt += cnt
            
    importances = {'f_cnt': f_cnt,
                   'r_cnt': r_cnt,
                   'importance_f': importance_f,
                   'importance_r': importance_r}
    torch.save(importances, f"./importances/{cfg.model_family}_{cfg.split}.pt")

if __name__ == "__main__":
    main()
