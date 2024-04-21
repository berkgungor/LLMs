import os
from copy import deepcopy
from random import randrange
from functools import partial
import torch
import accelerate
import bitsandbytes as bnb
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model,
    PeftModel
)

import wandb
from trl import SFTTrainer
wandb.login()

model_name = "HuggingFaceH4/zephyr-7b-beta"
new_model = "zephyr-Golf-Instruct-beta_v8"

### LOAD DATASET

dataset_dir = "//home////Repository//AI_Coach//golf_coach_dataset_v4.json"
dataset = load_dataset('json', data_files=dataset_dir, split='train')


def format_prompt(sample):

    INTRO = "You are an personal assistant for  practice. help them to improve their  skills and enlighten them about  techniques."
    END = "Instruction: Write appropriate response to the user's question and memorize the conversation."
    
    conversations = ""
    for response in sample["conversations"]:
      from_, value = response["from"], response["value"]
      conversations += f"<{from_}>: " + value + "\n"

    sample["text"] = f"{INTRO}\n\n{conversations}"

    return sample


dataset = dataset.map(
    format_prompt,
    remove_columns=["conversations"]
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
)

model.config.use_cache = False
model.config.pretraining_tp = 1
model.gradient_checkpointing_enable()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.padding_side = 'right'
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = True
tokenizer.add_bos_token, tokenizer.add_eos_token

model = prepare_model_for_kbit_training(model)
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['up_proj', 'base_layer', 'down_proj']
)
model = get_peft_model(model, peft_config)

#=========================== WANDB ==========================================
run = wandb.init(
    project="finetuning_zephyr7b_8",
    name="log_dataset",
)

dataset.save_to_disk("GolfInstruct_prep8.hf")
artifact = wandb.Artifact(name="GolfInstruct_prep8", type="dataset")
artifact.add_dir("./GolfInstruct_prep8.hf", name="train")
run.log_artifact(artifact)

run.finish()

run = wandb.init(
    project="finetuning_zephyr7b_8",   # Project name.
    name="run0",                     # name of the run within this project.
)
os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # Log model checkpoints.


#=========================== TRAINING ==========================================


#Hyperparamter
training_arguments = TrainingArguments(
    output_dir="./results_zephyr_v8",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=100,
    logging_steps=100,
    learning_rate=1e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=500,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="wandb"
)


# Setting sft parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    max_seq_length=512,
    dataset_text_field="text",
    tokenizer=tokenizer,
    args=training_arguments,
    packing=False,
)

"""trainer.train()
trainer.model.save_pretrained(new_model)
run.finish()
"""

### Merge model

def merge_model():
    # Merge the model with LoRA weights
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    merged_model= PeftModel.from_pretrained(base_model, new_model)
    merged_model= merged_model.merge_and_unload()
    torch.cuda.empty_cache()
    # Save the merged model
    merged_model.save_pretrained("zephyr_beta_merged_model_v8",safe_serialization=True)
    tokenizer.save_pretrained("zephyr_beta_merged_model_v8")
    print("========= merged Model saved =========")
    return merged_model, tokenizer

merge_model()
